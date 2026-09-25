import torch
import torch.nn as nn
from infrastructure import BaseModule, ParameterSchema
from graph_constructor import GraphConstructor
from gat_layers import HeterogeneousGATv2Layer, JumpingKnowledge


class Module2(BaseModule):
    def __init__(self, schema: ParameterSchema):
        super().__init__(schema)
        self.graph_constructor = GraphConstructor(schema)

        input_dim = max(
            2,
            schema.num_formants + schema.phoneme_embedding_dim,
            schema.num_harmonics + schema.num_noise_bands
        )

        self.gat_layer1 = HeterogeneousGATv2Layer(input_dim, 64, num_heads=4)
        self.gat_layer2 = HeterogeneousGATv2Layer(64, 64, num_heads=4)
        self.gat_layer3 = HeterogeneousGATv2Layer(64, 64, num_heads=4)

        self.jk = JumpingKnowledge(mode='cat')

        self.anomaly_mlp = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.node_projection = nn.ModuleDict({
            'source': nn.Linear(2, input_dim),
            'filter': nn.Linear(schema.num_formants + schema.phoneme_embedding_dim, input_dim),
            'resonator': nn.Linear(schema.num_harmonics + schema.num_noise_bands, input_dim)
        })

    def forward(self, parameters: dict) -> dict:
        batch_size = parameters['f0'].shape[0]
        num_frames = parameters['f0'].shape[1]

        all_anomalies = []

        for b in range(batch_size):
            batch_params = {
                'f0': parameters['f0'][b:b+1],
                'glottal_tilt': parameters['glottal_tilt'][b:b+1],
                'formant_bandwidths': parameters['formant_bandwidths'][b:b+1],
                'phoneme_context': parameters['phoneme_context'][b:b+1],
                'harmonic_amplitudes': parameters['harmonic_amplitudes'][b:b+1],
                'noise_magnitude': parameters['noise_magnitude'][b:b+1]
            }

            graph = self.graph_constructor.build_graph(batch_params)

            x_dict = {}
            for node_type in ['source', 'filter', 'resonator']:
                x_dict[node_type] = self.node_projection[node_type](graph[node_type].x)

            edge_index_dict = {}
            for edge_type in graph.edge_types:
                edge_index_dict[edge_type] = graph[edge_type].edge_index

            h1 = self.gat_layer1(x_dict, edge_index_dict)
            h2 = self.gat_layer2(h1, edge_index_dict)
            h3 = self.gat_layer3(h2, edge_index_dict)

            h_jk = {}
            for node_type in ['source', 'filter', 'resonator']:
                h_jk[node_type] = self.jk([h1[node_type], h2[node_type], h3[node_type]])

            anomalies = {}
            for node_type in ['source', 'filter', 'resonator']:
                anomalies[node_type] = self.anomaly_mlp(h_jk[node_type])

            all_anomalies.append(anomalies)

        stacked_anomalies = {}
        for node_type in ['source', 'filter', 'resonator']:
            stacked_anomalies[node_type] = torch.stack([a[node_type] for a in all_anomalies], dim=0)

        return {
            'anomalies': stacked_anomalies,
            'attention_weights': []
        }

    def get_output_shape(self, input_shape: tuple) -> dict:
        batch_size = input_shape[0]
        num_frames = input_shape[1]
        return {
            'anomalies_source': (batch_size, num_frames, 1),
            'anomalies_filter': (batch_size, num_frames, 1),
            'anomalies_resonator': (batch_size, num_frames, 1),
        }