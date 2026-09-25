import pytest
import torch
from infrastructure import ParameterSchema
from module4 import Module4


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
def module4(schema):
    return Module4(schema)


@pytest.fixture
def test_inputs(schema):
    batch_size = 2
    num_frames = schema.num_frames

    module2_output = {
        'anomalies': {
            'source': torch.rand(batch_size, num_frames, 1),
            'filter': torch.rand(batch_size, num_frames, 1),
            'resonator': torch.rand(batch_size, num_frames, 1)
        }
    }

    module3_output = torch.randn(batch_size, num_frames, 128)

    return module2_output, module3_output


def test_module4_initialization(module4, schema):
    assert module4.schema == schema
    assert module4.tcn_block1 is not None
    assert module4.classification_head is not None
    assert module4.explainability_head is not None


def test_module4_forward_shape(module4, test_inputs, schema):
    module2_output, module3_output = test_inputs
    output = module4(module2_output, module3_output)

    assert 'p_fake' in output
    assert 'explanation' in output

    batch_size = module3_output.shape[0]

    assert output['p_fake'].shape == (batch_size, 1)
    assert output['explanation'].shape == (batch_size, 2)


def test_module4_output_ranges(module4, test_inputs):
    module2_output, module3_output = test_inputs
    output = module4(module2_output, module3_output)

    assert torch.all(output['p_fake'] >= 0)
    assert torch.all(output['p_fake'] <= 1)

    assert torch.all(output['explanation'] >= 0)
    assert torch.all(output['explanation'] <= 1)

    explanation_sum = output['explanation'].sum(dim=-1)
    assert torch.allclose(explanation_sum, torch.ones_like(explanation_sum), atol=1e-5)


def test_module4_gradient_flow(module4, test_inputs):
    module2_output, module3_output = test_inputs

    module3_output.requires_grad_(True)

    output = module4(module2_output, module3_output)

    loss = output['p_fake'].mean()
    loss.backward()

    assert module3_output.grad is not None


def test_module4_different_batch_sizes(module4, schema):
    num_frames = schema.num_frames

    for batch_size in [1, 4]:
        module2_output = {
            'anomalies': {
                'source': torch.rand(batch_size, num_frames, 1),
                'filter': torch.rand(batch_size, num_frames, 1),
                'resonator': torch.rand(batch_size, num_frames, 1)
            }
        }
        module3_output = torch.randn(batch_size, num_frames, 128)

        output = module4(module2_output, module3_output)

        assert output['p_fake'].shape[0] == batch_size
        assert output['explanation'].shape[0] == batch_size


def test_module4_get_output_shape(module4, schema):
    shapes = module4.get_output_shape((2,))

    assert 'p_fake' in shapes
    assert 'explanation' in shapes
    assert shapes['p_fake'][0] == 2
    assert shapes['explanation'][0] == 2