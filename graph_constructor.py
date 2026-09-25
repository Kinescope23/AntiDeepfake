import torch
from torch_geometric.data import HeteroData
from infrastructure import ParameterSchema


class GraphConstructor:
    def __init__(self, schema: ParameterSchema):
        self.schema = schema
        self.edge_types = [
            ('source', 'excites', 'filter'),
            ('filter', 'filters', 'resonator'),
            ('source', 'pressures', 'resonator'),
            ('source', 'inertia', 'source'),
            ('filter', 'inertia', 'filter'),
            ('resonator', 'inertia', 'resonator')
        ]

    def build_graph(self, parameters: dict) -> HeteroData:
        num_frames = parameters['f0'].shape[1]
        data = HeteroData()

        data['source'].x = torch.cat([
            parameters['f0'],
            parameters['glottal_tilt']
        ], dim=-1).squeeze(0)

        data['filter'].x = torch.cat([
            parameters['formant_bandwidths'],
            parameters['phoneme_context']
        ], dim=-1).squeeze(0)

        data['resonator'].x = torch.cat([
            parameters['harmonic_amplitudes'],
            parameters['noise_magnitude']
        ], dim=-1).squeeze(0)

        data['source', 'excites', 'filter'].edge_index = self._build_intra_frame_edges(num_frames)
        data['filter', 'filters', 'resonator'].edge_index = self._build_intra_frame_edges(num_frames)
        data['source', 'pressures', 'resonator'].edge_index = self._build_intra_frame_edges(num_frames)

        data['source', 'inertia', 'source'].edge_index = self._build_inter_frame_edges(num_frames)
        data['filter', 'inertia', 'filter'].edge_index = self._build_inter_frame_edges(num_frames)
        data['resonator', 'inertia', 'resonator'].edge_index = self._build_inter_frame_edges(num_frames)

        return data

    def _build_intra_frame_edges(self, num_frames: int) -> torch.Tensor:
        source_indices = list(range(num_frames))
        target_indices = list(range(num_frames))
        return torch.tensor([source_indices, target_indices], dtype=torch.long)

    def _build_inter_frame_edges(self, num_frames: int) -> torch.Tensor:
        source_indices = list(range(num_frames - 1))
        target_indices = list(range(1, num_frames))
        return torch.tensor([source_indices, target_indices], dtype=torch.long)