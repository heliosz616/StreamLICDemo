"""Export quantized model tensors to a NumPy archive."""
import argparse
from pathlib import Path

import numpy as np
import torch


def flatten_tensors(value, prefix=''):
    result = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, dict):
        for key, item in value.items():
            name = str(key) if not prefix else prefix + '.' + str(key)
            nested = flatten_tensors(item, name)
            if result.keys() & nested.keys():
                raise ValueError('Duplicate tensor name in model archive')
            result.update(nested)
    return result


def export_parameters(model_path, output_path):
    archive = torch.load(str(model_path), map_location='cpu', weights_only=False)
    tensors = flatten_tensors(archive)
    if not tensors:
        raise ValueError('Model archive contains no tensors')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez(stream, **tensors)
    return len(tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    count = export_parameters(args.model, args.output)
    print(f'Exported {count} tensors to {args.output}')


if __name__ == '__main__':
    main()
