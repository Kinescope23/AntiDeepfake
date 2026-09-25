import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class PhonemeExtractor(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for param in self.model.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(768, schema.phoneme_embedding_dim)

    def forward(self, audio):
        with torch.no_grad():
            outputs = self.model(audio)
            hidden_states = outputs.last_hidden_state
        return self.projection(hidden_states)