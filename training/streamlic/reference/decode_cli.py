"""Decode a StreamLIC arithmetic bitstream and reconstruct a PNG image."""

import argparse
import json
from pathlib import Path
import time

from streamlic.entropy.bitstream import FrameBitstream
from streamlic.entropy.cdf import CDFBundle
from streamlic.model.integer_model import IntegerStreamLIC, save_image_tensor
from streamlic.reference.codec import decode_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a .slc bitstream and reconstruct its RGB PNG")
    parser.add_argument("--bitstream", required=True, help="input .slc file")
    parser.add_argument("--model", required=True, help="quantized model archive")
    parser.add_argument("--cdf", required=True, help="binary CDF bundle used by the encoder")
    parser.add_argument("--output", required=True, help="reconstructed PNG output")
    args = parser.parse_args()

    started = time.monotonic()
    bitstream_path = Path(args.bitstream)
    data = bitstream_path.read_bytes()
    frame = FrameBitstream.from_bytes(data)

    model = IntegerStreamLIC.load(args.model)
    cdfs = CDFBundle.load(args.cdf)
    _, _, _, _, reconstruction = decode_frame(model, data, cdfs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_tensor(reconstruction, output_path)

    summary = {
        "bitstream": str(bitstream_path),
        "width": frame.width,
        "height": frame.height,
        "cdf_precision": 24,
        "z_symbols": frame.width // 64 * (frame.height // 64) * 96,
        "y_symbols": frame.width // 16 * (frame.height // 16) * 192,
        "z_payload_bytes": len(frame.z_payload),
        "y_payload_bytes": sum(map(len, frame.y_payloads)),
        "bitstream_bytes": len(data),
        "output": str(output_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
