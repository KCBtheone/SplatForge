"""Export the actual optimized MILo surface, without a post-hoc TSDF."""

import torch

from ..mesh.io import write_mesh


class MILoExporter:
    @torch.no_grad()
    def export(self, pipeline, path, config):
        if not pipeline.mesh_state.active:
            raise ValueError('MILo export requires an optimized mesh state')
        mesh = pipeline.mesh_state(pipeline.model.means, pipeline.model.quats,
            pipeline.model.params['log_scales'])
        if not len(mesh.faces):
            raise ValueError('MILo optimized zero level set is empty; checkpoint is preserved')
        write_mesh(path, mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy())
        return {'exporter': 'milo-optimized-mesh', 'vertices': len(mesh.vertices),
            'faces': len(mesh.faces), 'pivot_gaussians': len(pipeline.mesh_state.pivot_ids),
            'tetrahedra': len(pipeline.mesh_state.cells), 'step': pipeline.completed_step,
            'topology_backend': pipeline.config.milo.topology_backend, 'path': str(path)}
