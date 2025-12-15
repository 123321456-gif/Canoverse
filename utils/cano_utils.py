import trimesh
import numpy as np



def normalize_mesh_to_unit_sphere(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize the input mesh to fit within a unit sphere centered at the origin.
    """
    vertices = mesh.vertices.copy()
    center = vertices.mean(axis=0)
    vertices -= center
    scale = (vertices**2).sum(axis=1).max()**0.5
    vertices /= scale
    mesh.vertices = vertices
    return mesh
