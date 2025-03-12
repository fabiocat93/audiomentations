import json
import pytest

import numpy as np

from audiomentations import AddGaussianSNR


class TestGaussianSNR:
    def test_gaussian_noise_snr_defaults(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(p=1.0)
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        assert samples_out.dtype == np.float32
        assert not (float(std_out) == pytest.approx(0.0))
        assert std_out > std_in

    def test_gaussian_noise_snr(self):
        np.random.seed(42)
        samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
        augmenter = AddGaussianSNR(min_snr_db=15, max_snr_db=35, p=1.0)
        std_in = np.mean(np.abs(samples_in))
        samples_out = augmenter(samples=samples_in, sample_rate=16000)
        std_out = np.mean(np.abs(samples_out))
        assert samples_out.dtype == np.float32
        assert not (float(std_out) == pytest.approx(0.0))
        assert std_out > std_in

    def test_serialize_parameters(self):
        np.random.seed(42)
        transform = AddGaussianSNR(min_snr_db=15, max_snr_db=35, p=1.0)
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        # NumPy 2.0 enforces stricter type handling, and np.float32 is no longer automatically converted to a native Python float when serializing JSON
        json.dumps(transform.serialize_parameters(), default=lambda o: float(o) if isinstance(o, np.generic) else o)

    def test_gaussian_noise_snr_multichannel(self):
        np.random.seed(42)
        samples = np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32)
        augmenter = AddGaussianSNR(min_snr_db=15, max_snr_db=35, p=1.0)
        samples_out = augmenter(samples=samples, sample_rate=16000)

        assert samples_out.dtype == np.float32
        assert float(np.sum(np.abs(samples_out))) > float(np.sum(np.abs(samples)))

    def test_validation(self):
        with pytest.raises(ValueError):
            AddGaussianSNR(
                min_snr_db=40.0,
                max_snr_db=20.0,
            )
