import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from infrastructure import ParameterSchema
from module1 import Module1


@pytest.fixture
def schema():
    return ParameterSchema(
        sample_rate=16000,
        duration_seconds=1.0,
        frame_hop_size=256,
        stft_window_size=1024,
        num_harmonics=64,
        num_formants=4,
        num_noise_bands=65,
        phoneme_embedding_dim=256
    )


@pytest.fixture
def module1(schema):
    with patch("module1.PhonemeExtractor") as MockPhoneme:
        mock_instance = MagicMock(spec=nn.Module)
        mock_instance.return_value = torch.randn(2, 63, 256)
        mock_instance.__call__ = lambda x: torch.randn(x.shape[0], 63, 256)
        MockPhoneme.return_value = mock_instance
        module = Module1(schema)
        module.phoneme_extractor = mock_instance
        return module


@pytest.fixture
def test_audio(schema):
    return torch.randn(2, schema.num_samples)


def test_module1_initialization(module1, schema):
    assert module1.schema == schema
    assert module1.audio_processor is not None
    assert module1.backbone is not None
    assert module1.heads is not None


def test_module1_forward_shape(module1, test_audio, schema):
    output = module1(test_audio)

    assert "f0" in output
    assert "harmonic_amplitudes" in output
    assert "glottal_tilt" in output
    assert "formant_bandwidths" in output
    assert "noise_magnitude" in output
    assert "phoneme_context" in output

    expected_frames = schema.num_frames

    assert output["f0"].shape == (2, expected_frames, 1)
    assert output["harmonic_amplitudes"].shape == (2, expected_frames, schema.num_harmonics)
    assert output["glottal_tilt"].shape == (2, expected_frames, 1)
    assert output["formant_bandwidths"].shape == (2, expected_frames, schema.num_formants)
    assert output["noise_magnitude"].shape == (2, expected_frames, schema.num_noise_bands)
    assert output["phoneme_context"].shape == (2, expected_frames, schema.phoneme_embedding_dim)


def test_module1_physical_constraints(module1, test_audio):
    output = module1(test_audio)

    assert torch.all(output["f0"] > 0)
    assert torch.all(output["glottal_tilt"] >= 0) and torch.all(output["glottal_tilt"] <= 1)
    assert torch.all(output["formant_bandwidths"] > 0)
    assert torch.all(output["noise_magnitude"] > 0)

    harmonic_sums = output["harmonic_amplitudes"].sum(dim=-1)
    assert torch.allclose(harmonic_sums, torch.ones_like(harmonic_sums), atol=1e-4)


def test_module1_gradient_flow(module1, test_audio):
    test_audio.requires_grad_(True)
    output = module1(test_audio)

    loss = output["f0"].mean()
    loss.backward()

    assert test_audio.grad is not None
    assert test_audio.grad.shape == test_audio.shape


def test_module1_audio_processor(module1, test_audio):
    processed = module1.audio_processor(test_audio)

    assert "audio_normalized" in processed
    assert "log_spectrogram" in processed
    assert processed["audio_normalized"].shape == test_audio.shape
    assert processed["log_spectrogram"].shape[0] == 2
    assert processed["log_spectrogram"].shape[1] == module1.schema.num_stft_bins


def test_module1_different_batch_sizes(module1, schema):
    for batch_size in [1, 4]:
        audio = torch.randn(batch_size, schema.num_samples)
        output = module1(audio)

        assert output["f0"].shape[0] == batch_size
        assert output["harmonic_amplitudes"].shape[0] == batch_size


def test_module1_get_output_shape(module1, schema):
    shapes = module1.get_output_shape((2, schema.num_samples))

    assert shapes["f0"] == (2, schema.num_frames, 1)
    assert shapes["harmonic_amplitudes"] == (2, schema.num_frames, schema.num_harmonics)
    assert shapes["phoneme_context"] == (2, schema.num_frames, schema.phoneme_embedding_dim)