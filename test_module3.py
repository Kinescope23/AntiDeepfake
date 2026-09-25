import pytest
import torch
from infrastructure import ParameterSchema
from module3 import Module3


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
def module3(schema):
    return Module3(schema)


@pytest.fixture
def test_inputs(schema):
    batch_size = 2
    num_frames = schema.num_frames
    num_freq_bins = schema.num_stft_bins

    spectrogram = torch.randn(batch_size, num_frames, num_freq_bins)

    parameters = {
        'f0': torch.rand(batch_size, num_frames, 1) * 200 + 80,
        'harmonic_amplitudes': torch.rand(batch_size, num_frames, schema.num_harmonics),
        'glottal_tilt': torch.rand(batch_size, num_frames, 1),
        'formant_bandwidths': torch.rand(batch_size, num_frames, schema.num_formants) * 100 + 50,
        'noise_magnitude': torch.rand(batch_size, num_frames, schema.num_noise_bands)
    }

    return spectrogram, parameters


def test_module3_initialization(module3, schema):
    assert module3.schema == schema
    assert module3.conv_init is not None
    assert module3.block1 is not None
    assert module3.pool1 is not None


def test_module3_forward_shape(module3, test_inputs, schema):
    spectrogram, parameters = test_inputs
    output = module3(spectrogram, parameters)

    batch_size = spectrogram.shape[0]
    num_frames = spectrogram.shape[1]

    assert output.shape == (batch_size, num_frames, 128)


def test_module3_gradient_flow(module3, test_inputs):
    spectrogram, parameters = test_inputs

    spectrogram.requires_grad_(True)
    for key in parameters:
        parameters[key].requires_grad_(True)

    output = module3(spectrogram, parameters)

    loss = output.mean()
    loss.backward()

    assert spectrogram.grad is not None
    assert spectrogram.grad.shape == spectrogram.shape

    for key in parameters:
        assert parameters[key].grad is not None


def test_module3_different_batch_sizes(module3, schema):
    num_frames = schema.num_frames
    num_freq_bins = schema.num_stft_bins

    for batch_size in [1, 4]:
        spectrogram = torch.randn(batch_size, num_frames, num_freq_bins)
        parameters = {
            'f0': torch.rand(batch_size, num_frames, 1),
            'harmonic_amplitudes': torch.rand(batch_size, num_frames, schema.num_harmonics),
            'glottal_tilt': torch.rand(batch_size, num_frames, 1),
            'formant_bandwidths': torch.rand(batch_size, num_frames, schema.num_formants),
            'noise_magnitude': torch.rand(batch_size, num_frames, schema.num_noise_bands)
        }

        output = module3(spectrogram, parameters)
        assert output.shape == (batch_size, num_frames, 128)


def test_module3_get_output_shape(module3, schema):
    shape = module3.get_output_shape((2, schema.num_frames, schema.num_stft_bins))
    assert shape == (2, schema.num_frames, 128)