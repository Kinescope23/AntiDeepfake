import torch
import torch.nn as nn
from infrastructure import BaseModule, AudioProcessor, ParameterSchema
from backbone import CRNNBackbone, ParameterHeads
from phoneme_extractor import PhonemeExtractor


class Module1(BaseModule):
    def __init__(self, schema: ParameterSchema):
        super().__init__(schema)
        self.audio_processor = AudioProcessor(schema)
        self.backbone = CRNNBackbone(schema)
        self.heads = ParameterHeads(schema)
        self.phoneme_extractor = PhonemeExtractor(schema)

    def forward(self, audio: torch.Tensor) -> dict:
        processed = self.audio_processor(audio)
        raw_audio = processed["audio_normalized"]
        spectrogram_shape = processed["log_spectrogram"].shape

        target_frames = spectrogram_shape[2]

        features = self.backbone(raw_audio)

        if features.shape[1] != target_frames:
            features = torch.nn.functional.interpolate(
                features.transpose(1, 2),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        parameters = self.heads(features)

        phonemes = self.phoneme_extractor(raw_audio)

        if phonemes.shape[1] != target_frames:
            phonemes = torch.nn.functional.interpolate(
                phonemes.transpose(1, 2),
                size=target_frames,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        parameters["phoneme_context"] = phonemes

        return parameters

    def get_output_shape(self, input_shape: tuple) -> dict:
        batch_size = input_shape[0]
        num_frames = self.schema.num_frames
        return {
            "f0": (batch_size, num_frames, 1),
            "harmonic_amplitudes": (batch_size, num_frames, self.schema.num_harmonics),
            "glottal_tilt": (batch_size, num_frames, 1),
            "formant_bandwidths": (batch_size, num_frames, self.schema.num_formants),
            "noise_magnitude": (batch_size, num_frames, self.schema.num_noise_bands),
            "phoneme_context": (batch_size, num_frames, self.schema.phoneme_embedding_dim),
        }