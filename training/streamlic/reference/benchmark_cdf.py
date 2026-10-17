import argparse
import json
import time

from streamlic.entropy.cdf import build_bundle_from_archive
from streamlic.model.integer_model import IntegerStreamLIC
from streamlic.reference.codec import encode_frame, psnr


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the fixed 24-bit StreamLIC CDF")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()
    model = IntegerStreamLIC.load(args.model)
    image = model.load_image(args.image, args.height, args.width)
    source = (image + 128).clamp(0, 255).to(dtype=model.parameters["g_a.0"]["weight"].dtype)
    reports = []
    for precision in (24,):
        started = time.monotonic()
        cdfs = build_bundle_from_archive(args.model, precision)
        encoded = encode_frame(model, image, cdfs)
        reports.append({
            "precision": precision,
            "bytes": len(encoded.data),
            "bits_per_pixel": 8.0 * len(encoded.data) / (args.width * args.height),
            "psnr_db": psnr(source, encoded.result.reconstruction),
            "z_clamped": encoded.result.z_clamped,
            "y_clamped": encoded.result.y_clamped,
            "seconds": time.monotonic() - started,
        })
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
