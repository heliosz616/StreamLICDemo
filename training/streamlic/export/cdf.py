import argparse
from pathlib import Path

from streamlic.entropy.cdf import build_bundle_from_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic StreamLIC CDF tables")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", type=int, default=24, choices=(16, 20, 24))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    bundle = build_bundle_from_archive(args.model, args.precision)
    bundle.save(args.output)
    summary = args.summary or str(Path(args.output).with_suffix(".json"))
    bundle.write_summary(summary)
    print("wrote {} y CDFs and {} z CDFs at precision {}".format(
        len(bundle.y), len(bundle.z), bundle.precision))
    print("y symbol bounds: {}..{}; z symbol bounds: {}..{}".format(
        min(t.minimum for t in bundle.y), max(t.maximum for t in bundle.y),
        min(t.minimum for t in bundle.z), max(t.maximum for t in bundle.z)))


if __name__ == "__main__":
    main()
