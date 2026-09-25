import torch
import torch.nn as nn
from infrastructure import ParameterSchema
from module1 import Module1
from module2 import Module2
from module3 import Module3
from module4 import Module4


class DeepfakeDetectorPipeline(nn.Module):
    def __init__(self, schema: ParameterSchema):
        super().__init__()
        self.schema = schema
        self.module1 = Module1(schema)
        self.module2 = Module2(schema)
        self.module3 = Module3(schema)
        self.module4 = Module4(schema)

    def forward(self, audio: torch.Tensor) -> dict:
        parameters = self.module1(audio)

        processed = self.module1.audio_processor(audio)
        log_spectrogram = processed["log_spectrogram"]
        spectrogram_transposed = log_spectrogram.transpose(1, 2)

        module2_output = self.module2(parameters)

        module3_output = self.module3(spectrogram_transposed, parameters)

        final_output = self.module4(module2_output, module3_output)

        return {
            'p_fake': final_output['p_fake'],
            'explanation': final_output['explanation'],
            'parameters': parameters,
            'module2_anomalies': module2_output['anomalies']
        }