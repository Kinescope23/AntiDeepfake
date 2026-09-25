import pytest
import torch
from infrastructure import ParameterSchema
from module2 import Module2


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
def module2(schema):
    return Module2(schema)


@pytest.fixture
def test_parameters(schema):
    batch_size = 2
    num_frames = schema.num_frames

    return {
        'f0': torch.rand(batch_size, num_frames, 1) * 200 + 80,
        'glottal_tilt': torch.rand(batch_size, num_frames, 1),
        'formant_bandwidths': torch.rand(batch_size, num_frames, schema.num_formants) * 100 + 50,
        'phoneme_context': torch.randn(batch_size, num_frames, schema.phoneme_embedding_dim),
        'harmonic_amplitudes': torch.rand(batch_size, num_frames, schema.num_harmonics),
        'noise_magnitude': torch.rand(batch_size, num_frames, schema.num_noise_bands)
    }


def test_module2_initialization(module2, schema):
    assert module2.schema == schema
    assert module2.graph_constructor is not None
    assert module2.gat_layer1 is not None
    assert module2.anomaly_mlp is not None


def test_module2_forward_shape(module2, test_parameters, schema):
    output = module2(test_parameters)

    assert 'anomalies' in output
    assert 'attention_weights' in output

    batch_size = test_parameters['f0'].shape[0]
    num_frames = test_parameters['f0'].shape[1]

    assert 'source' in output['anomalies']
    assert 'filter' in output['anomalies']
    assert 'resonator' in output['anomalies']

    assert output['anomalies']['source'].shape == (batch_size, num_frames, 1)
    assert output['anomalies']['filter'].shape == (batch_size, num_frames, 1)
    assert output['anomalies']['resonator'].shape == (batch_size, num_frames, 1)


def test_module2_anomaly_range(module2, test_parameters):
    output = module2(test_parameters)

    for node_type in ['source', 'filter', 'resonator']:
        anomalies = output['anomalies'][node_type]
        assert torch.all(anomalies >= 0)
        assert torch.all(anomalies <= 1)


def test_module2_gradient_flow(module2, test_parameters):
    for key in test_parameters:
        test_parameters[key].requires_grad_(True)

    output = module2(test_parameters)

    total_loss = 0
    for node_type in ['source', 'filter', 'resonator']:
        total_loss = total_loss + output['anomalies'][node_type].mean()

    total_loss.backward()

    for key in test_parameters:
        assert test_parameters[key].grad is not None


def test_module2_different_batch_sizes(module2, schema):
    for batch_size in [1, 3]:
        num_frames = schema.num_frames
        parameters = {
            'f0': torch.rand(batch_size, num_frames, 1) * 200 + 80,
            'glottal_tilt': torch.rand(batch_size, num_frames, 1),
            'formant_bandwidths': torch.rand(batch_size, num_frames, schema.num_formants) * 100 + 50,
            'phoneme_context': torch.randn(batch_size, num_frames, schema.phoneme_embedding_dim),
            'harmonic_amplitudes': torch.rand(batch_size, num_frames, schema.num_harmonics),
            'noise_magnitude': torch.rand(batch_size, num_frames, schema.num_noise_bands)
        }

        output = module2(parameters)
        assert output['anomalies']['source'].shape[0] == batch_size


def test_module2_get_output_shape(module2, schema):
    shapes = module2.get_output_shape((2, schema.num_frames, 1))

    assert 'anomalies_source' in shapes
    assert shapes['anomalies_source'][0] == 2
    assert shapes['anomalies_source'][1] == schema.num_frames