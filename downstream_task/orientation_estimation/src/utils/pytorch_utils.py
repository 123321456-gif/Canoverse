

import torch
import numpy as np
import os

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
    )
from pytorch3d.io import load_objs_as_meshes

# 因为编码问题，pytorch读取不了的mesh，自己写一个读取代码
def read_obj_file(obj_file_path):
    verts = []
    faces = []
    
    with open(obj_file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            parts = line.split()
            if len(parts) > 0:
                if parts[0] == 'v':  # Vertex definition
                    # Convert parts[1:], which are vertex coordinates, to floats
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':  # Face definition
                    # Convert parts[1:], which are references to vertex indices, to integers
                    # Note: .obj files are 1-indexed, so we adjust by subtracting 1 for 0-indexed Python lists
                    faces.append([int(x.split('/')[0]) - 1 for x in parts[1:]])

    # Convert lists to numpy arrays
    verts_np = np.array(verts, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)
    
    # Convert numpy arrays to torch tensors
    verts_tensor = torch.from_numpy(verts_np)
    faces_tensor = torch.from_numpy(faces_np)
    
    return verts_tensor, faces_tensor

'''# 重写 obj 文件，（仅替换verts），这样还能有纹理
def rewrite_objFile(obj_path:str, new_v:torch.Tensor, save_path:str=None):
    """读取OBJ文件，替换顶点数据并保存为新文件。

    Args:
        obj_path (str): obj文件的路径 
        new_v (torch.Tensor): 替换掉顶点的数据 new_v.shape = (n, 3)

    Returns:
        _type_: _description_
    """
    with open(obj_path, 'r') as file:
        lines = file.readlines()
    
    v_count = 0  # 用于跟踪替换的顶点数量
    for i, line in enumerate(lines):
        if line.startswith('v '):
            if v_count < len(new_v):
                vertex_data = 'v ' + ' '.join(map(str, new_v[v_count].tolist())) + '\n'
                lines[i] = vertex_data # new_v[v_count] + '\n'  # 确保有换行符
            v_count += 1
    
    if save_path is not None:
        with open(save_path, 'w') as file:
            file.writelines(lines)
    
    return lines
'''
# @profile
def rewrite_objFile(obj_path: str, new_v: torch.Tensor, save_path: str = None):
    """read obj file, replace vertex data and save as new file.

    Args:
        obj_path (str): the path of obj file
        new_v (torch.Tensor): the new verts used to repace the old verts new_v.shape = (n, 3)

    Returns:
        _type_: _description_
    """
    with open(obj_path, 'r') as file:
        lines = file.readlines()
    
    new_v = new_v.cpu().numpy() 
    vertex_lines = (f"v {v[0]} {v[1]} {v[2]}\n" for v in new_v)


    # 判断新点的个数和原始点的个数是否相同，不同则报错
    vertex_count = sum(1 for line in lines if line.startswith('v '))
    if vertex_count != len(new_v):
        raise ValueError(f"Number of vertices in new_v ({len(new_v)}) does not match number of vertices in {obj_path} ({vertex_count})")

    lines = [
        next(vertex_lines) if line.startswith('v ') else line 
        for i, line in enumerate(lines)
    ]
    # lines = [
    #     next(vertex_lines) if line.startswith('v ') and i < len(new_v) else line 
    #     for i, line in enumerate(lines)
    # ]

    if save_path is not None:
        with open(save_path, 'w') as file:
            file.writelines(lines)
    
    return lines

# add texture to pytorch mesh
def add_texture_to_mesh(mesh, verts_colors=None):
    """给pytorch3d mesh加纹理

    Args:
        mesh (_type_): pytorch3d 类型的mesh
        verts_colors (_type_): tensor类型， verts_num * 3

    Returns:
        _type_: 加了纹理的mesh
    """
    device = mesh.device
    # 创建一个灰白色的纹理图
    verts_packed = mesh.verts_packed()  # 获取网格的顶点

    if verts_colors is None:
        # 创建灰白色的顶点颜色
        verts_colors = torch.full(verts_packed.shape, 0.5, device=device)

    # 将颜色赋给mesh
    textures = TexturesVertex(verts_features=verts_colors[None])
    vis_mesh = Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()], textures=textures)
    return vis_mesh

# add transformation to mesh
def transformation_mesh(mesh, pose:torch.tensor, scale:torch.tensor=None)->Meshes:
    """
    mesh:pytorch3D 中的
    pose: 4*4
    """
    # 1. 取出pytorch3d mesh 中的顶点
    vertices = mesh.verts_packed()
    if scale is not None:
        # 2. 对顶点进行尺寸变换
        vertices = vertices * scale
    # 3. 对顶点进行位姿变换
    vertices = torch.matmul(vertices, pose[:3,:3].T) + pose[:3,3]
    # 4. 新建一个mesh
    new_mesh = Meshes(verts=[vertices], faces=[mesh.faces_packed()], textures=mesh.textures)

    return new_mesh

# rotate the mesh
def rotate_mesh(mesh, rot:torch.tensor)->Meshes:
    """
    mesh:pytorch3D 中的
    pose: 3*3
    """
    # 1. 取出pytorch3d mesh 中的顶点
    vertices = mesh.verts_packed()
    # 3. 对顶点进行位姿变换
    vertices = torch.matmul(vertices, rot.T) 
    # 4. 新建一个mesh
    new_mesh = Meshes(verts=[vertices], faces=[mesh.faces_packed()], textures=mesh.textures)

    return new_mesh

# 找到mesh面片和pic像素的对应关系
def find_pix_to_face(mesh, view_idx, device, camera_distance=2.2)->torch.Tensor:
    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    view = views[view_idx]
    R, T = look_at_view_transform(camera_distance, view[0], view[1], device=device)
    # 位姿，渲染，光照的设置
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)
    raster_settings = RasterizationSettings(
        image_size=800, 
        blur_radius=0.0,
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    pix_to_face = rasterizer(mesh).pix_to_face  # 这个就是取像素和面片对应关系的函数
    pix_to_face = pix_to_face[0]
    # 取出深度图
    depth = rasterizer(mesh).zbuf
    
    return pix_to_face, depth

# read obj file and backup fast pt file
def read_obj_file_and_backup(file_path:str, device:torch.device)->Meshes:
    """
    读取obj文件，如果读取失败，就用自己的读取方式
    """
    objs_dirs = os.path.dirname(file_path)  # /data4/jl/datasets/DREDS/cad_model/real_cat_known/aeroplane/0
    objs_name = os.path.basename(file_path).split('.')[0]  # model.obj
    try:
        meshes = load_objs_as_meshes([file_path], device=device)
    except:
        print('objs_path:', file_path)
        verts, faces = read_obj_file(file_path)
        verts = verts.to(device)
        faces = faces.to(device)
        textures = TexturesVertex(verts_features=torch.ones_like(verts).unsqueeze(0))
        textures = textures.to(device)
        meshes = Meshes(verts=[verts], faces=[faces], textures=textures)
    vertices = meshes.verts_packed()
    faces = meshes.faces_packed()
    textures = meshes.textures
    # save back up file
    torch.save(vertices, objs_dirs+'/'+objs_name+'_verts.pt')
    torch.save(faces, objs_dirs+'/'+objs_name+'_faces.pt')
    torch.save(textures, objs_dirs+'/'+objs_name+'_textures.pt')
    return meshes


if __name__ == '__main__':
    # test rewrite_objFile
    obj_path = 'data/interim/objaverse_dataset/newObj/airplane/glbs/ins_0000/026b36fa6cfe4800b65534a5a5d5bfa6.obj'
    verts, faces = read_obj_file(obj_path)
    print(verts.shape)
    new_v = verts * 0.5
    rewrite_objFile(obj_path, new_v, save_path='results/unit_test/test.obj')