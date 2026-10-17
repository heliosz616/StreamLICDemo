#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
host_binary="${STREAMLIC_HOST_BINARY:-${script_dir}/../runtime/bin/streamlic_encoder_host}"
xrt_setup="${STREAMLIC_XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
device_bdf="${STREAMLIC_DEVICE_BDF:-}"
device_index=0
thermal_limit_c="${STREAMLIC_THERMAL_LIMIT_C:-87}"
timeout_seconds="${STREAMLIC_BENCHMARK_TIMEOUT_SECONDS:-120}"
reset_after=1
frame_count=64
slot_count=4
batch_size=8
codec_workers=4
dump_raw_first=0
dump_raw_all=0
analysis_only=0
analysis_signatures=0
reload_analysis_each_frame=0
verify_input=0
poison_outputs=0
xclbin=
parameters=
cdf=
input_dir=
output_dir=

while (($#)); do
  case "$1" in
    --xclbin) xclbin=$2; shift 2 ;;
    --parameters) parameters=$2; shift 2 ;;
    --cdf) cdf=$2; shift 2 ;;
    --input-dir) input_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --device) device_index=$2; shift 2 ;;
    --device-bdf) device_bdf=$2; shift 2 ;;
    --count) frame_count=$2; shift 2 ;;
    --slots) slot_count=$2; shift 2 ;;
    --batch-size) batch_size=$2; shift 2 ;;
    --codec-workers) codec_workers=$2; shift 2 ;;
    --dump-raw-first) dump_raw_first=1; shift ;;
    --dump-raw-all) dump_raw_all=1; shift ;;
    --analysis-only) analysis_only=1; shift ;;
    --analysis-signatures) analysis_signatures=1; shift ;;
    --reload-analysis-each-frame) reload_analysis_each_frame=1; shift ;;
    --verify-input) verify_input=1; shift ;;
    --poison-outputs) poison_outputs=1; shift ;;
    --timeout-seconds) timeout_seconds=$2; shift 2 ;;
    --keep-programmed) reset_after=0; shift ;;
    *) echo "unknown or incomplete option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$xclbin" || -z "$parameters" || -z "$cdf" ||
      -z "$input_dir" || -z "$output_dir" ]]; then
  echo "usage: $0 --xclbin FILE --parameters DIR --cdf FILE --input-dir DIR --output-dir DIR [--count N] [--device N] [--device-bdf BDF] [--slots N] [--batch-size 1..16] [--codec-workers N] [--dump-raw-first|--dump-raw-all] [--analysis-only] [--analysis-signatures] [--reload-analysis-each-frame] [--verify-input] [--poison-outputs] [--timeout-seconds N] [--keep-programmed]" >&2
  exit 2
fi
if [[ ! "$frame_count" =~ ^[1-9][0-9]*$ || ! "$device_index" =~ ^[0-9]+$ ]]; then
  echo "frame count must be a positive integer" >&2
  exit 2
fi
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ||
      ! "$slot_count" =~ ^[1-9][0-9]*$ ||
      ! "$batch_size" =~ ^[1-9][0-9]*$ || "$batch_size" -gt 16 ||
      ! "$codec_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "timeout, slots, and workers must be positive; batch size must be 1..16" >&2
  exit 2
fi
for required in "$host_binary" "$xrt_setup" "$xclbin" "$cdf"; do
  [[ -f "$required" ]] || { echo "required file not found: $required" >&2; exit 2; }
done
[[ -d "$parameters" && -d "$input_dir" ]] || {
  echo "parameter or input directory not found" >&2
  exit 2
}

mapfile -t images < <(find "$input_dir" -maxdepth 1 -type f \
  \( -iname '*.ppm' -o -iname '*.png' \) -print | LC_ALL=C sort | head -n "$frame_count")
if ((${#images[@]} != frame_count)); then
  echo "found ${#images[@]} PNG/PPM images, expected $frame_count" >&2
  exit 2
fi
mkdir -p "$output_dir"
timestamp=$(date +%Y%m%dT%H%M%S)
run_log="$output_dir/encoder_${frame_count}_${timestamp}.log"
sensor_log="$output_dir/sensors_${frame_count}_${timestamp}.csv"
abort_marker="$output_dir/.thermal_abort_${timestamp}"
timeout_marker="$output_dir/.timeout_abort_${timestamp}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
# XRT 2022.1 setup is not nounset-clean.
set +u
source "$xrt_setup" >/dev/null
set -u
command -v xbutil >/dev/null || {
  echo "xbutil is unavailable after sourcing $xrt_setup" >&2
  exit 2
}

if [[ -z "$device_bdf" ]]; then
  hwmon_candidates=()
  for candidate in /sys/bus/pci/devices/*/hwmon/*; do
    if [[ -r "$candidate/temp4_input" && -r "$candidate/temp10_input" &&
          -r "$candidate/power1_input" ]]; then
      hwmon_candidates+=("$candidate")
    fi
  done
  if ((${#hwmon_candidates[@]} == 1)); then
    device_bdf=$(basename "$(dirname "$(dirname "${hwmon_candidates[0]}")")")
  elif ((${#hwmon_candidates[@]} == 0)); then
    echo "no U50 hwmon device was found; pass --device-bdf BDF" >&2
    exit 2
  else
    echo "multiple FPGA hwmon devices were found; pass --device-bdf BDF" >&2
    exit 2
  fi
fi

hwmon_dir=$(find "/sys/bus/pci/devices/${device_bdf}/hwmon" -mindepth 1 -maxdepth 1 -type d -print -quit)
[[ -n "$hwmon_dir" && -r "$hwmon_dir/temp4_input" &&
   -r "$hwmon_dir/temp10_input" && -r "$hwmon_dir/power1_input" ]] || {
  echo "U50 hwmon sensors are unavailable for $device_bdf" >&2
  exit 2
}
initial_fpga_millic=$(<"$hwmon_dir/temp4_input")
initial_hbm_millic=$(<"$hwmon_dir/temp10_input")
if ((initial_fpga_millic >= thermal_limit_c * 1000)); then
  echo "refusing to start: FPGA is already at ${initial_fpga_millic} millidegrees C" >&2
  exit 3
fi
printf 'preflight_fpga_c=%d.%03d preflight_hbm_c=%d.%03d\n' \
  "$((initial_fpga_millic / 1000))" "$((initial_fpga_millic % 1000))" \
  "$((initial_hbm_millic / 1000))" "$((initial_hbm_millic % 1000))"

host_args=(--xclbin "$xclbin" --parameters "$parameters" --cdf "$cdf"
           --device "$device_index"
           --output-dir "$output_dir"
           --slots "$slot_count" --batch-size "$batch_size"
           --codec-workers "$codec_workers")
if ((dump_raw_first)); then host_args+=(--dump-raw-first); fi
if ((dump_raw_all)); then host_args+=(--dump-raw-all); fi
if ((analysis_only)); then host_args+=(--analysis-only); fi
if ((analysis_signatures)); then host_args+=(--analysis-signatures); fi
if ((reload_analysis_each_frame)); then host_args+=(--reload-analysis-each-frame); fi
if ((verify_input)); then host_args+=(--verify-input); fi
if ((poison_outputs)); then host_args+=(--poison-outputs); fi
for image in "${images[@]}"; do host_args+=(--image "$image"); done

echo "timestamp,fpga_c,hbm_c,power_w" >"$sensor_log"
"$host_binary" "${host_args[@]}" >"$run_log" 2>&1 &
host_pid=$!
deadline_epoch=$(($(date +%s) + timeout_seconds))
(
  terminate_host() {
    kill -TERM "$host_pid" 2>/dev/null || return 0
    for _ in {1..25}; do
      kill -0 "$host_pid" 2>/dev/null || return 0
      sleep 0.2
    done
    echo "host did not stop after SIGTERM; sending SIGKILL" >>"$run_log"
    kill -KILL "$host_pid" 2>/dev/null || true
  }
  while kill -0 "$host_pid" 2>/dev/null; do
    fpga_millic=$(<"$hwmon_dir/temp4_input")
    hbm_millic=$(<"$hwmon_dir/temp10_input")
    power_microw=$(<"$hwmon_dir/power1_input")
    printf '%s,%d.%03d,%d.%03d,%d.%06d\n' "$(date +%s.%N)" \
      "$((fpga_millic / 1000))" "$((fpga_millic % 1000))" \
      "$((hbm_millic / 1000))" "$((hbm_millic % 1000))" \
      "$((power_microw / 1000000))" "$((power_microw % 1000000))" >>"$sensor_log"
    if ((fpga_millic >= thermal_limit_c * 1000)); then
      echo "thermal guard reached ${fpga_millic} millidegrees C" >>"$run_log"
      : >"$abort_marker"
      terminate_host
      break
    fi
    if (( $(date +%s) >= deadline_epoch )); then
      echo "benchmark timeout reached ${timeout_seconds} seconds" >>"$run_log"
      : >"$timeout_marker"
      terminate_host
      break
    fi
    sleep 0.2
  done
) &
monitor_pid=$!

host_status=0
wait "$host_pid" || host_status=$?
wait "$monitor_pid" || true
reset_status=0
if ((reset_after)); then
  echo "resetting U50 $device_bdf after encoder run" >>"$run_log"
  xbutil reset --device "$device_bdf" --force >>"$run_log" 2>&1 || reset_status=$?
fi
cat "$run_log"
echo "sensor_log=$sensor_log"
if [[ -f "$abort_marker" ]]; then
  echo "benchmark stopped by ${thermal_limit_c} C thermal guard" >&2
  exit 3
fi
if [[ -f "$timeout_marker" ]]; then
  echo "benchmark stopped by ${timeout_seconds}-second timeout" >&2
  exit 4
fi
if ((reset_status)); then
  echo "U50 reset failed with status $reset_status" >&2
  exit 5
fi
exit "$host_status"
