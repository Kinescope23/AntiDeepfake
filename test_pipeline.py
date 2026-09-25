import pytest
import torch
import torch.nn as nn
from infrastructure import ParameterSchema
from pipeline import DeepfakeDetectorPipeline


class MockPhonemeExtractor(nn.Module):
    def __init__(self, schema: ParameterSchema):
        super().__init__()
        self.schema = schema

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size = audio.shape[0]
        num_frames = self.schema.num_frames
        return torch.randn(batch_size, num_frames, self.schema.phoneme_embedding_dim)


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
def pipeline(schema):
    pipe = DeepfakeDetectorPipeline(schema)
    pipe.module1.phoneme_extractor = MockPhonemeExtractor(schema)
    return pipe


@pytest.fixture
def test_audio(schema):
    return torch.randn(2, schema.num_samples)


def test_pipeline_initialization(pipeline, schema):
    assert pipeline.schema == schema
    assert pipeline.module1 is not None
    assert pipeline.module2 is not None
    assert pipeline.module3 is not None
    assert pipeline.module4 is not None


def test_pipeline_forward_shape(pipeline, test_audio, schema):
    output = pipeline(test_audio)

    assert 'p_fake' in output
    assert 'explanation' in output
    assert 'parameters' in output
    assert 'module2_anomalies' in output

    batch_size = test_audio.shape[0]

    assert output['p_fake'].shape == (batch_size, 1)
    assert output['explanation'].shape == (batch_size, 2)


def test_pipeline_output_ranges(pipeline, test_audio):
    output = pipeline(test_audio)

    assert torch.all(output['p_fake'] >= 0)
    assert torch.all(output['p_fake'] <= 1)

    assert torch.all(output['explanation'] >= 0)
    assert torch.all(output['explanation'] <= 1)

    explanation_sum = output['explanation'].sum(dim=-1)
    assert torch.allclose(explanation_sum, torch.ones_like(explanation_sum), atol=1e-5)


def test_pipeline_gradient_flow(pipeline, test_audio):
    test_audio.requires_grad_(True)

    output = pipeline(test_audio)

    loss = output['p_fake'].mean()
    loss.backward()

    assert test_audio.grad is not None
    assert test_audio.grad.shape == test_audio.shape


def test_pipeline_different_batch_sizes(pipeline, schema):
    for batch_size in [1, 4]:
        audio = torch.randn(batch_size, schema.num_samples)
        output = pipeline(audio)

        assert output['p_fake'].shape[0] == batch_size
        assert output['explanation'].shape[0] == batch_size