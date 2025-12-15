
"""检查3D分割结果和对应的方向是否正确

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""
# 这个文件用于check 3D semantic detection，semantic dir的中间结果

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
import open3d as o3d

from src.utils.pytorch_utils import add_texture_to_mesh
from src.utils.vis import vis_pytorch3d_mesh, display_pts_with_arrow, get_colors, plot_2d_line
from src.seg3d.seg3d import calculate_weighted_direction_vectors_tensor
# from src.seg3d.seg3d_utils import mask_normalization, semantic_2D_to_3d, cluster_confidences_tensor


def draw_point_cloud_and_ray(points, ray_origin, ray_direction, ray_length=5, coordinate=None):
    """
    绘制点云和一条射线。

    参数:
    - points: 点云数据，可以是NumPy数组或Tensor，形状为(N, 3)。
    - ray_origin: 射线的起点，可以是NumPy数组或Tensor，形状为(3,)。
    - ray_direction: 射线的方向，可以是NumPy数组或Tensor，形状为(3,)。
    - ray_length: 射线的长度，默认为5。
    """
    # 如果输入数据是Tensor，则转换为NumPy数组
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(ray_origin, torch.Tensor):
        ray_origin = ray_origin.numpy()
    if isinstance(ray_direction, torch.Tensor):
        ray_direction = ray_direction.numpy()

    # 创建点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 计算射线的终点
    ray_end = ray_direction # ray_origin + ray_length * ray_direction / np.linalg.norm(ray_direction)

    # 创建射线作为线段
    lines = [[ray_origin, ray_end]]
    print('lines:', lines)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([ray_origin, ray_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    if coordinate is not None:
        # 坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # 可视化点云和射线
        o3d.visualization.draw_geometries([point_cloud, line_set, coordinate_frame])
    else:
        # 可视化点云和射线
        o3d.visualization.draw_geometries([point_cloud, line_set])

def get_colors2(semantic_confi):
    semantic_confi = semantic_confi.numpy()
    # 创建自定义颜色映射
    # cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # 将置信度映射到颜色
    norm = plt.Normalize(vmin=np.min(semantic_confi), vmax=np.max(semantic_confi))
    colors = cmap(norm(semantic_confi))
    return colors


def visualize(support_mesh, muti_views_verts_colors, parts_num, device):
        
    normalized_verts_colors = muti_views_verts_colors  


    # normalized_verts_colors中第一列大于0.5的索引， semantic_color_mesh对应索引那一行的第0个给1
    semantic_color_mesh = torch.zeros_like(normalized_verts_colors, device=device)
    for lie in range(parts_num):
        semantic_confi = normalized_verts_colors[:,lie]  
        semantic_color_mesh = torch.tensor(get_colors(semantic_confi, cmap='RdBu'))
        # print(semantic_color_mesh.shape)
        # asdf
    
        vis_mesh = add_texture_to_mesh(support_mesh, semantic_color_mesh)
        vis_pytorch3d_mesh(vis_mesh)

# 将语义转为方向向量，numpy版
def calculate_weighted_direction_vectors(semantics, verts):

    # 初始化权重中心数组
    weighted_centers = np.zeros((semantics.shape[1], 3))
    # 初始化总权重数组
    total_weights = np.zeros(semantics.shape[1])
    
    # 计算每个标签的权重中心
    for i in range(semantics.shape[1]):
        # 对于第i个标签，计算每个点的权重（概率）
        weights = semantics[:, i]
        # 累加权重
        total_weights[i] = np.sum(weights)
        # 如果总权重不为零，计算加权中心
        if total_weights[i] > 0:
            weighted_centers[i] = np.sum(verts * weights[:, np.newaxis], axis=0) / total_weights[i]
    return weighted_centers
# # 将语义转为方向向量，高斯版本
# def calculate_weighted_direction_vectors(semantics, verts):

#     # 初始化权重中心数组
#     weighted_centers = np.zeros((semantics.shape[1], 3))
#     # 初始化总权重数组
#     total_weights = np.zeros(semantics.shape[1])
    
#     # 计算每个标签的权重中心
#     for i in range(semantics.shape[1]):
#         # 对于第i个标签，计算每个点的权重（概率）
#         weights = semantics[:, i]
#         # 累加权重
#         total_weights[i] = np.sum(weights)
#         # 如果总权重不为零，计算加权中心
#         if total_weights[i] > 0:
#             weighted_centers[i] = np.average(verts, axis=0, weights=weights)
#     return weighted_centers

def vis_semantic3d(data_path):
     

    # mesh semantic可视化
    verts = np.load(data_path+'support_verts.npy')
    faces = np.load(data_path+'support_faces.npy')
    semantics = np.load(data_path+'semantics.npy')
    # 语义分布可视化
    mesh = Meshes(verts=[torch.tensor(verts, dtype=torch.float32)], faces=[torch.tensor(faces, dtype=torch.int64)])

    '''# debug 尝试获得uv map
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    aux = mesh.verts_packed_to_mesh_vert_idx()
    V, _ = verts
    verts_to_uv_index = torch.zeros(V, dtype=torch.int64)
    verts_to_uv_index[faces.verts_index.flatten()]=faces.textures_index.flatten()  # <- This is where the carelessness is
    verts_to_uvs = aux.verts_uvs[verts_to_uv_index]
    print('verts_to_uvs:', verts_to_uvs.shape)
    asdf'''


    visualize(mesh, torch.tensor(semantics, dtype=torch.float32), semantics.shape[1], 'cpu')

    # # 画第3列的折线图
    # plot_2d_line(semantics[:, 2])  # 原本是0，聚类后变成重心点了
    # asdf

    # semantic 转dirs
    dirs = calculate_weighted_direction_vectors_tensor(semantics, verts)
    dirs = dirs.cpu().numpy()
    print('dirs:', dirs)    
    # 计算xoz平面上的模长
    dirs_xoz = dirs[:, [0,2]]
    norms = np.linalg.norm(dirs_xoz, axis=1)
    print('norms:', norms)
 
    # dirs = calculate_weighted_direction_vectors(semantics, verts)
    # semantic dir 可视化
    for dir in dirs:
        # direction_vectors = np.array([0,1,1])
        # 打印dir的模长
        # print('dir:', dir)
        # print('norm:', np.linalg.norm(dir))
        ray_origin = np.array([0,0,0])
        display_pts_with_arrow(verts, ray_origin, dir)

        # draw_point_cloud_and_ray(verts, ray_origin, dir)

    # # 可视化一个平面上的点云
    # display_plane_with_pts(verts)




if __name__ == '__main__':
    
    # data_path = "/data4/LJ/workstation/PartSLIP2/results/0506-shapeNet/chair/results/41fd861b4f4b8baf3adc3470b30138f3_random0/rot5/semantic3d/"
    # data_path = 'results/objaverse/newObj/results/antenna/ins_0000/rot_0/semantic3d/'
    # data_path = 'results/render_test/semantic3d/'

    # # objaverse 快速找到路径
    # category = 'booklet'
    # insName = 'ins_0002'
    # data_path = 'results/objaverse/newObj/results/'+category+'/'+insName+'/rot_0/semantic3d/'
    # # temp
    # category = 'booklet'
    # insName = 'ins_0000'
    # data_path = 'results/objaverse/temp/results/'+category+'/'+insName+'/rot_0/semantic3d/'

    # data_path = 'results/objaverse/newObj/results/sword/ins_0009/rot_0/semantic3d/'
    # data_path = 'results/objaverse/temp/results/baseball_bat/ins_0000/rot_0/semantic3d/'
    


    # data_path = '/data4/jl/project/one-shot/results/objaverse_lvis/newObj/results/shark/ins_0001/rot_0/semantic3d/'

    # data_path = '/data4/jl/project/one-shot/results/objaverse_lvis/newObj/results/car_(automobile)/ins_0000/rot_0/semantic3d/'

    # data_path = '/data4/jl/project/one-shot/results_org/objaverse_lvis/temp/results/chair/ins_0000/rot_0/semantic3d/'
    data_path = 'results/775bd20947464914b6f77175580c63a9.obj/rot_0/semantic3d/'

    vis_semantic3d(data_path)

