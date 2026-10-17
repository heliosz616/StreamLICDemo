import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from streamlic.entropy.cdf import CDFBundle, build_bundle_from_archive
from streamlic.model.integer_model import IntegerStreamLIC, save_image_tensor, tensor_hwc_flat
from streamlic.reference.codec import decode_frame_for_test, encode_frame, psnr


def _write_golden(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_patch = torch.nn.functional.pixel_unshuffle(
        torch.nn.functional.pad(result.input_tensor, (1, 3, 1, 1)), 2)
    np.savetxt(output_dir / "test_patch.txt",
               image_patch.permute(0, 2, 3, 1).contiguous().view(-1).numpy(), fmt="%d")
    np.savetxt(output_dir / "test_y.txt", tensor_hwc_flat(result.y), fmt="%d")
    np.savetxt(output_dir / "test_golden1.txt", tensor_hwc_flat(result.z_symbols), fmt="%d")
    interleaved = torch.empty((1, 384, result.y_symbols.shape[2], result.y_symbols.shape[3]),
                              dtype=torch.int32)
    interleaved[:, 0::2] = result.y_symbols
    interleaved[:, 1::2] = result.y_indexes
    np.savetxt(output_dir / "test_golden2.txt", tensor_hwc_flat(interleaved), fmt="%d")
    np.savetxt(output_dir / "test_hyper.txt", tensor_hwc_flat(result.hyper), fmt="%d")
    np.savetxt(output_dir / "test_y_hat.txt", tensor_hwc_flat(result.y_hat), fmt="%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exact StreamLIC reference encoder")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--output", required=True, help="encoded .slc output")
    parser.add_argument("--reconstruction")
    parser.add_argument("--golden-dir")
    parser.add_argument("--verify-decode", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    model = IntegerStreamLIC.load(args.model)
    cdfs = CDFBundle.load(args.cdf) if args.cdf else build_bundle_from_archive(
        args.model, 24)
    image = model.load_image(args.image, args.height, args.width)
    encoded = encode_frame(model, image, cdfs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded.data)
    if args.reconstruction:
        save_image_tensor(encoded.result.reconstruction, args.reconstruction)
    if args.golden_dir:
        _write_golden(encoded.result, Path(args.golden_dir))

    verified = False
    if args.verify_decode:
        decoded = decode_frame_for_test(model, encoded.data, cdfs)
        z, y, indexes, y_hat, reconstruction = decoded
        comparisons = {
            "z": torch.equal(z, encoded.result.z_symbols),
            "y_symbols": torch.equal(y, encoded.result.y_symbols),
            "y_indexes": torch.equal(indexes, encoded.result.y_indexes),
            "y_hat": torch.equal(y_hat, encoded.result.y_hat),
            "reconstruction": torch.equal(reconstruction, encoded.result.reconstruction),
        }
        if not all(comparisons.values()):
            raise RuntimeError("reference decode mismatch: {}".format(comparisons))
        verified = True

    source_u8 = (image + 128).clamp(0, 255).to(torch.uint8)
    summary = {
        "width": args.width, "height": args.height,
        "precision": cdfs.precision, "bytes": len(encoded.data),
        "entropy_streams": 4,
        "bits_per_pixel": 8.0 * len(encoded.data) / (args.width * args.height),
        "psnr_db": psnr(source_u8, encoded.result.reconstruction),
        "z_symbols": int(encoded.result.z_symbols.numel()),
        "y_symbols": int(encoded.result.y_symbols.numel()),
        "z_clamped": encoded.result.z_clamped,
        "y_clamped": encoded.result.y_clamped,
        "verified_decode": verified,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
