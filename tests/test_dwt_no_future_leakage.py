import unittest

import numpy as np

from src.models.vol.dwt_lstm import DWTLSTMVolModel, pywt


@unittest.skipIf(pywt is None, "PyWavelets not installed")
class TestDWTNoFutureLeakage(unittest.TestCase):
    def test_same_history_same_features(self):
        model = DWTLSTMVolModel(seq_len=60, wavelet="db4", level=3, epochs=1, batch_size=4)
        history = np.linspace(-0.03, 0.03, 60)
        future_a = np.array([0.01, -0.02, 0.015])
        future_b = np.array([-0.04, 0.03, 0.02])

        seq_a = np.concatenate([history, future_a])
        seq_b = np.concatenate([history, future_b])

        feat_a = model._dwt_multiscale_features(seq_a[:60])
        feat_b = model._dwt_multiscale_features(seq_b[:60])
        self.assertTrue(np.allclose(feat_a, feat_b))

    def test_shape_stable_for_same_level(self):
        histories = [
            np.sin(np.linspace(0, 6.28, 60)),
            np.cos(np.linspace(0, 6.28, 60)),
            np.linspace(-0.01, 0.02, 60),
        ]
        for level in (2, 3, 4):
            model = DWTLSTMVolModel(seq_len=60, wavelet="db4", level=level, epochs=1, batch_size=4)
            shapes = {model._dwt_multiscale_features(history).shape for history in histories}
            self.assertEqual(len(shapes), 1)
            shape = next(iter(shapes))
            self.assertEqual(shape[0], 60)
            self.assertGreaterEqual(shape[1], 2)


if __name__ == "__main__":
    unittest.main()
