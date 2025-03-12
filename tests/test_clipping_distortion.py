import numpy as np
import pytest

from audiomentations import ClippingDistortion, Compose


class TestClippingDistortion:
    def test_distort(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                ClippingDistortion(
                    min_percentile_threshold=20, max_percentile_threshold=40, p=1.0
                )
            ]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        # Since NumPy 2.0 defaults more operations to float64
        assert samples_out.dtype in (np.float32, np.float64)
        assert len(samples_out) == sample_len
        assert sum(abs(samples_out)) < sum(abs(samples_in))

    def test_distort_multichannel(self):
        sample_len = 32000
        samples_in = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = ClippingDistortion(
            min_percentile_threshold=20, max_percentile_threshold=40, p=1.0
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        # Since NumPy 2.0 defaults more operations to float64
        assert samples_out.dtype in (np.float32, np.float64)
        assert samples_out.shape == samples_in.shape
        assert np.sum(np.abs(samples_out)) < np.sum(np.abs(samples_in))
        assert np.amax(samples_out[0, :]) == pytest.approx(np.amax(samples_out[1, :]))
