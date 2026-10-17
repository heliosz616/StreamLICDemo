from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from streamlic.entropy.cdf import CDFBundle, build_bundle_from_archive
from streamlic.export.parameters import export_parameters, flatten_tensors
from streamlic.model.integer_model import IntegerStreamLIC
from streamlic.reference.cli import _write_golden
from streamlic.reference.codec import encode_frame, decode_frame_for_test

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/quant_weights_and_pdf.pt'


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cdf_export_matches_runtime(self):
        generated = build_bundle_from_archive(MODEL, 24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cdf.bin'
            generated.save(path)
            self.assertEqual(path.read_bytes(), (ROOT / 'runtime/parameters/cdf_p24.bin').read_bytes())

    def test_parameter_export(self):
        archive = torch.load(MODEL, map_location='cpu', weights_only=False)
        expected = flatten_tensors(archive)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parameters.npz'
            self.assertEqual(export_parameters(MODEL, path), len(expected))
            with np.load(path, allow_pickle=False) as actual:
                self.assertEqual(set(actual.files), set(expected))
                for name, value in expected.items():
                    np.testing.assert_array_equal(actual[name], value)
                    self.assertEqual(actual[name].dtype, value.dtype)

    def test_reference_round_trip_and_goldens(self):
        model = IntegerStreamLIC.load(MODEL)
        cdfs = CDFBundle.load(ROOT / 'runtime/parameters/cdf_p24.bin')
        image = model.load_image(ROOT / 'examples/test_pattern.png', 128, 128)
        encoded = encode_frame(model, image, cdfs)
        decoded = decode_frame_for_test(model, encoded.data, cdfs)
        expected = (encoded.result.z_symbols, encoded.result.y_symbols,
                    encoded.result.y_indexes, encoded.result.y_hat,
                    encoded.result.reconstruction)
        for actual, wanted in zip(decoded, expected):
            self.assertTrue(torch.equal(actual, wanted))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _write_golden(encoded.result, path)
            self.assertEqual(len(list(path.glob('*.txt'))), 6)
            np.testing.assert_array_equal(
                np.loadtxt(path / 'test_y.txt', dtype=np.int32),
                encoded.result.y.permute(0, 2, 3, 1).contiguous().view(-1).numpy())


if __name__ == '__main__':
    unittest.main()
