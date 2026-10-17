import contextlib
import io
import unittest

import torch
from streamlic_q import ARLiteSparse_Q


class ModelSmokeTest(unittest.TestCase):
    def test_training_modes_and_export(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        with contextlib.redirect_stdout(io.StringIO()):
            model = ARLiteSparse_Q().train()
            model.toggle_quantization(False)
            model.toggle_mask(False)
            x = torch.zeros(1, 3, 64, 64)
            result = model(x)
            self.assertEqual(tuple(result['x_hat'].shape), tuple(x.shape))
            loss = result['x_hat'].square().mean()
            for likelihood in result['likelihoods'].values():
                loss = loss - likelihood.clamp_min(1e-9).log().mean()
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
            model.set_all_mask()
            model.toggle_mask(True)
            model.toggle_quantization(True)
            self.assertTrue(torch.isfinite(model(x)['x_hat']).all().item())
            model.fix_encoder_flag(True)
            self.assertTrue(torch.isfinite(model(x)['x_hat']).all().item())
            model.init_pdf_table()
            model.toggle_pdf_training_flag(True)
            self.assertTrue(torch.isfinite(model(x)['x_hat']).all().item())
            params = model.export_quantized_parameters()
            self.assertTrue(params)
            self.assertTrue(any(k.startswith('g_a.') for k in params))


if __name__ == '__main__':
    unittest.main()
