import torch
import torch.nn as nn
from infrastructure import BaseModule, ParameterSchema
from tcn_block import TCNBlock


class Module4(BaseModule):
    def __init__(self, schema: ParameterSchema):
        super().__init__(schema)

        # 3 (anomalies) + 128 (texture features) = 131 channels
        self.tcn_block1 = TCNBlock(131, 128, kernel_size=3, dilation=1)
        self.tcn_block2 = TCNBlock(128, 128, kernel_size=3, dilation=2)
        self.tcn_block3 = TCNBlock(128, 128, kernel_size=3, dilation=4)
        self.tcn_block4 = TCNBlock(128, 128, kernel_size=3, dilation=8)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classification_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.explainability_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, module2_output: dict, module3_output: torch.Tensor) -> dict:
        anomalies_source = module2_output['anomalies']['source']
        anomalies_filter = module2_output['anomalies']['filter']
        anomalies_resonator = module2_output['anomalies']['resonator']

        batch_size, num_frames, _ = anomalies_source.shape

        sp_features = torch.cat([
            anomalies_source,
            anomalies_filter,
            anomalies_resonator
        ], dim=-1)

        texture_features = module3_output

        combined = torch.cat([sp_features, texture_features], dim=-1)

        x = combined.permute(0, 2, 1)

        x = self.tcn_block1(x)
        x = self.tcn_block2(x)
        x = self.tcn_block3(x)
        x = self.tcn_block4(x)

        x = self.global_pool(x).squeeze(-1)

        p_fake = self.classification_head(x)
        explanation = self.explainability_head(x)

        return {
            'p_fake': p_fake,
            'explanation': explanation
        }

    def get_output_shape(self, input_shape: tuple) -> dict:
        batch_size = input_shape[0]
        return {
            'p_fake': (batch_size, 1),
            'explanation': (batch_size, 2)
        }