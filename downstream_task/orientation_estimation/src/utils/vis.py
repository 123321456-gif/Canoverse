import open3d as o3d
import numpy as np
import torch
import time
import os
from PIL import Image
import math
import trimesh

from typing import Literal

##########################trimesh################################
# 保存带着纹理得obj
def save_mesh_as_obj_with_texture(mesh, semantics:np.array, obj_path):
    """
    将pytorch3d类型的mesh保存为带有纹理的OBJ文件

    参数:
    mesh: Meshes对象
    semantics: 纹理数据 (torch.Tensor)
    obj_path: 保存OBJ文件的路径
    texture_path: 保存纹理图片的路径
    """
    # 获取顶点和面
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()

    # 获取顶点颜色
    # verts_colors =   # semantics.cpu().numpy()
    verts_colors = semantics# np.random.rand(verts.shape[0], 3)

    # 创建trimesh对象
    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=verts_colors)
    # 可视化
    trimesh_mesh.show()
    # asdf
    # 保存为OBJ文件
    obj_data = trimesh.exchange.export.export_obj(trimesh_mesh, include_texture=True)
    
    # 确保obj_data是一个字典
    if isinstance(obj_data, tuple):
        obj, data = obj_data
    else:
        obj = obj_data
        data = {}

    with open(obj_path, 'w') as f:
        f.write(obj)
    
    # 保存MTL文件和纹理图片
    for k, v in data.items():
        with open(os.path.join(os.path.dirname(obj_path), k), 'wb') as f:
            f.write(v)
    
    # 重新加载
    trimesh_mesh = trimesh.load(obj_path)
    # 可视化
    trimesh_mesh.show()

#############################open3d 相关可视化函数################################
# 可视化点云在一个平面上
def display_plane_with_pts(point_cloud1):
    """
    Display two point clouds together using Open3D.

    :param point_cloud1: Numpy array of shape (n, 3) representing the first point cloud.
    :param point_cloud2: Numpy array of shape (n, 3) representing the second point cloud.
    """
    if torch.is_tensor(point_cloud1):
        if point_cloud1.is_cuda:
            point_cloud1 = point_cloud1.to('cpu').detach().numpy()
            point_cloud2 = point_cloud2.to('cpu').detach().numpy()
        else:
            point_cloud1 = point_cloud1.numpy()
            point_cloud2 = point_cloud2.numpy()

    # Convert numpy arrays to Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)

    # Combine the point clouds for visualization
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])

    # 坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # pcd_combined = [pcd1, coordinate_frame]  #FOR1, 

    # 平面
    # 计算点云中Y的最小值
    min_y = np.min(point_cloud1[:, 1])
    plane = create_mesh_plane(center=[0, min_y, 0], normal=[0, 1, 0], extent=5.0)

    pcd_combined = [pcd1, plane,coordinate_frame]  #FOR1, 
    # Visualize the point clouds
    o3d.visualization.draw_geometries(pcd_combined)
# 建一个面片，方便在open3d中可视化
def create_mesh_plane(center, normal, extent=1.0):
    """
    创建一个指定中心和法线的平面网格。
    center: 平面中心点
    normal: 平面法线
    extent: 平面的大小
    """
    # 创建一个平面网格
    mesh = o3d.geometry.TriangleMesh.create_box(width=extent, height=0.01, depth=extent)
    mesh.translate(-mesh.get_center())
    
    # 将平面旋转到正确的方向
    # 首先计算旋转向量
    rotation_axis = np.cross([0, 1, 0], normal)
    rotation_angle = np.arccos(np.dot([0, 1, 0], normal))
    if rotation_angle == 0 or rotation_angle>3.141:
        rotation_matrix = np.eye(3)
    else:
        # 使用Rodrigues公式计算旋转矩阵
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    mesh.rotate(rotation_matrix, center=np.array([0, 0, 0]))
    
    # 将平面移动到指定的中心点
    mesh.translate(center)
    # print('plane: ', center, rotation_matrix, rotation_axis, rotation_angle)
    
    return mesh

# 可视化两个点云为不同的颜色
def display_two_point_clouds(point_cloud1, point_cloud2):
    """
    Display two point clouds together using Open3D.

    :param point_cloud1: Numpy array of shape (n, 3) representing the first point cloud.
    :param point_cloud2: Numpy array of shape (n, 3) representing the second point cloud.
    """
    if torch.is_tensor(point_cloud1):
        if point_cloud1.is_cuda:
            point_cloud1 = point_cloud1.to('cpu').detach().numpy()
            point_cloud2 = point_cloud2.to('cpu').detach().numpy()
        else:
            point_cloud1 = point_cloud1.numpy()
            point_cloud2 = point_cloud2.numpy()

    # Convert numpy arrays to Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
    # 每个点云一种颜色
    pcd1.paint_uniform_color([0, 0, 1])  # Red color
    pcd2.paint_uniform_color([1, 0, 0])  # Blue color
    # Combine the point clouds for visualization
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])

    # # 坐标系
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # pcd_combined = [pcd1, pcd2, coordinate_frame]  #FOR1, 

    pcd_combined = [pcd1, pcd2]  #FOR1, 
    # Visualize the point clouds
    o3d.visualization.draw_geometries(pcd_combined)

# 创建一个open3d的箭头
def create_arrow(start_point, direction, length=1.0, color=[1, 0, 0]):
    """
    Create an arrow in 3D space starting from `start_point` in the direction of `direction`.

    Parameters:
    start_point (np.ndarray): The starting point of the arrow.
    direction (np.ndarray): The direction vector of the arrow.
    length (float): The length of the arrow. Default is 1.0.
    color (list): The color of the arrow. Default is red.
    """
    direction = direction / np.linalg.norm(direction) * length
    end_point = start_point + direction
    
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.04,
        cylinder_height=0.8 * length,
        cone_height=0.2 * length
    )
    
    arrow.translate(start_point)
    arrow.paint_uniform_color(color)
    
    # Align arrow with the direction vector
    arrow_direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, arrow_direction)
    angle = np.arccos(np.dot(z_axis, arrow_direction))
    
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(rotation_matrix, center=start_point)
    
    return arrow
# 可视化点云和箭头
def display_pts_with_arrow(point_cloud:np.array, arrow_start:np.array, arrow_direction:np.array):
    """
    Display a point cloud and an arrow together using Open3D.

    :param point_cloud: Numpy array of shape (n, 3) representing the point cloud.
    :param arrow_start: Numpy array of shape (3,) representing the starting point of the arrow.
    :param arrow_direction: Numpy array of shape (3,) representing the direction of the arrow.
    """
    if torch.is_tensor(point_cloud):
        if point_cloud.is_cuda:
            point_cloud = point_cloud.to('cpu').detach().numpy()
        else:
            point_cloud = point_cloud.numpy()

    # Convert numpy arrays to Open3D point clouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Create the arrow
    arrow = create_arrow(arrow_start, arrow_direction, length=1.0, color=[1, 0, 0])

    # Create a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Visualize the arrow and point cloud
    o3d.visualization.draw_geometries([arrow, pcd, coordinate_frame])

# 将点云可视化为不同的颜色
def display_colored_point_cloud(pointClouds:np.array, pointColors:np.array):
    """_summary_

    Args:
        pointClouds (np.array): n,3 点云
        pointColors (np.array): n,3 每个点对应的rgb颜色, 范围:0-1
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointClouds)
    point_cloud.colors = o3d.utility.Vector3dVector(pointColors)
    o3d.visualization.draw_geometries([point_cloud]) # 显示点云

#############################pytorch3d 相关可视化函数################################
# 根据置信度将点云可视化为不同的颜色
from matplotlib import colormaps
def get_colors(values, cmap: Literal['viridis', 'RdBu'] = 'RdBu') -> np.ndarray:
    ratios = (values - values.min()) / (values.max() - values.min())
    return colormaps[cmap](ratios.view(-1).cpu().numpy())[..., :3]

# pytorch3d mesh的可视化，如果没有颜色，就生成蓝色
def vis_pytorch3d_mesh(mesh):
    from pytorch3d.vis.plotly_vis import plot_scene
    # Render the plotly figure 
    fig = plot_scene({
        "subplot1": {
            "cow_mesh": mesh
        }
    })
    fig.show()



#############################matplotlib 相关可视化函数(面片还是这个库好点)################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 可视化点云和面片
def plot_faces_and_points(faces:list[list], points:torch.Tensor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制面片
    for face in faces:
        vertices = [list(v) for v in face]
        poly3d = [[vertices[j] for j in range(len(vertices))]]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

    # 绘制点云
    points_np = points.cpu().numpy()  # 转换为 numpy 数组
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], color='b', s=10)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()

# 画2D折现图
def plot_2d_line(y:np.array, x=None, title='2D Line Plot', xlabel='X', ylabel='Y'):
    plt.figure(figsize=(8, 6))
    if x is None:
        x = np.arange(len(y))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# 画2D散点图
def plot_2d_scatter(x:np.array, y:np.array, labs=None , title='2D Scatter Plot', xlabel='X', ylabel='Y', save_path=None):
    plt.figure(figsize=(8, 6))
    if labs is None:
        plt.scatter(x, y)
    else:
        plt.scatter(x, y, c=labs, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


#################################################plt.imshow()#################################################
def load_image(image_path):
    """加载图片并返回"""
    return Image.open(image_path)

# 将二维矩阵可视化为热力图
def visualize_heatmap(matrix, title='Heatmap', cmap='viridis'):
    """
    可视化矩阵的热力图。 ; 必须是2维的矩阵。

    参数:
    matrix (np.ndarray or torch.Tensor): 输入矩阵。
    title (str): 图像标题。
    cmap (str): 颜色映射方案，默认是 'viridis'。
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()  # 如果输入是 torch.Tensor，将其转换为 numpy 数组
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect='auto', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

# 拼接多张图片 拼成一行
def merge_images(*images_or_paths, spacing=10, output_pic_path=None):
    """
    将多张图片拼接成一张图片。可以接受图片路径或Image对象。

    :param images_or_paths: 要拼接的图片路径或Image对象
    :param spacing: 图片之间的间隔
    :param output_pic_path: 输出图片的路径
    """
    images = []
    # 判断images_or_paths是否是列表中的列表，如果不是，封装成列表
    if not all(isinstance(item, (list, tuple)) for item in images_or_paths):
        images_or_paths = [images_or_paths]
        
    # 处理输入，判断是图像路径还是Image对象
    for item in images_or_paths[0]:
        if isinstance(item, str) and os.path.exists(item):  # 如果是路径且文件存在
            images.append(Image.open(item))
        elif isinstance(item, Image.Image):  # 如果是Image对象
            images.append(item)
        else:
            raise ValueError(f"Unsupported input: {item}")

    # 获取所有图片的宽度和高度
    widths, heights = zip(*(img.size for img in images))

    # 计算合并后的图片宽度和高度
    total_width = sum(widths) + spacing * (len(images) - 1)
    max_height = max(heights)
    
    # 创建一个新的空白图片
    merged_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    # 将所有图片拼接到新图片上
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.size[1]) // 2  # 垂直居中
        merged_image.paste(img, (x_offset, y_offset))
        x_offset += img.size[0] + spacing

    # 如果指定了输出路径，则保存图片
    if output_pic_path:
        merged_image.save(output_pic_path)

    # # 使用matplotlib显示合并后的图片
    # plt.imshow(merged_image)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    '''# 创建一个窗口，用于显示图片并支持按 ESC 退出
    fig, ax = plt.subplots()
    ax.imshow(merged_image)
    ax.axis('off')  # 不显示坐标轴

    def on_key(event):
        if event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()'''
    return merged_image
# 拼成一个矩阵
def merge_images2matrix(images_or_paths, spacing=10, output_pic_path=None):
    """
    Merge multiple images into a single image arranged in a grid. Accepts image paths or Image objects.

    :param images_or_paths: The paths of images or Image objects to be merged
    :param spacing: The spacing between images
    :param output_pic_path: The output path for the merged image
    """
    images = []

    # Handle input, checking if it's an image path or an Image object
    for item in images_or_paths:
        if isinstance(item, str) and os.path.exists(item):  # If it's a valid path
            images.append(Image.open(item))
        elif isinstance(item, Image.Image):  # If it's an Image object
            images.append(item)
        else:
            raise ValueError(f"Unsupported input: {item}")

    # Get the width and height of all images
    widths, heights = zip(*(img.size for img in images))

    # Calculate the number of images
    num_images = len(images)

    # Calculate the number of rows and columns for the best fit grid layout
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Calculate the maximum width and height of a single grid cell
    max_width = max(widths)
    max_height = max(heights)

    # Calculate the total width and height of the merged image
    total_width = cols * max_width + (cols - 1) * spacing
    total_height = rows * max_height + (rows - 1) * spacing

    # Create a new blank image
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Place all images into the new image according to the grid layout
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x_offset = col * (max_width + spacing)
        y_offset = row * (max_height + spacing)

        # 打印出每个图片的位置 和对应的图片名字
        # print(f"Image {os.path.basename(images_or_paths[i])} at row {row+1}, column {col+1}")

        merged_image.paste(img, (x_offset, y_offset))

    # Save the image if an output path is specified
    if output_pic_path:
        merged_image.save(output_pic_path)

    return merged_image

# 可视化图片
def visualize_image(img:Image):
     # 创建一个窗口，用于显示图片并支持按 ESC 退出
    fig, ax = plt.subplots()
    ax.imshow(img)
    # 设置窗口大小
    fig.set_size_inches(6, 6)
    ax.axis('off')  # 不显示坐标轴

    # 初始化 cate_annotation
    cate_annotation = None
    def on_key(event):
        nonlocal cate_annotation  # 声明使用外部变量 cate_annotation
        if event.key == 'escape':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# 可视化柱状图
def bar_data(args, labels=None):
    """
    函数接收一个数据列表或数据列表的列表和一个可选的标签列表，绘制柱状图。
    
    参数:
    args: 一个包含数据值的列表或包含多个数据类别值的列表的列表。
    labels: 数据类别的标签列表，应与数据列表的数量相匹配。
    """
    # 检查args是否是列表中的列表，如果不是，封装成列表
    if not all(isinstance(item, (list, tuple)) for item in args):
        args = [args]  # 将单个列表封装成列表的列表

    # 展开列表
    data = [item for sublist in args for item in sublist] if any(isinstance(el, list) for el in args) else args
    
    # 如果没有提供标签，则使用默认标签
    if not labels:
        labels = [f'{i+1}' for i in range(len(data))]
    # 检查标签的数量是否与数据列表的数量相匹配
    if labels and len(labels) != len(data):
        raise ValueError("Labels length must match the number of data lists provided.")

    # # 根据数据的数量来确定图形的宽度，每个数据点分配固定宽度，最大宽度为16英寸
    # width_per_item = 0.5  # 每个数据点的宽度，单位为英寸
    # total_width = min(len(data) * width_per_item, 14)  # 限制最大宽度为14英寸
    total_width = 10

    # 设置一组柔和的颜色
    # colors = ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#fb8500']
    colors = ['#E7906D', '#6B93C5', '#4FAE92', '#F4C17A']

    # 创建柱状图
    plt.figure(figsize=(total_width, 6))
    plt.bar(labels, data, color=colors[:len(data)])

    # 添加标题和标签
    plt.title('Data Overview')
    plt.xlabel('Categories')
    plt.ylabel('Values')

    # 添加网格
    plt.grid(True, alpha=0.6) # , linestyle='--', alpha=0.6

    # # 显示数值在柱状图上
    # for i, v in enumerate(data):
    #     plt.text(i, v + 1, str(v), ha='center', va='bottom')

    # 显示图形
    plt.show()
    '''# 示例调用
    total_aligned = 70
    total_noisy = 20
    total_unAligned = 10
    labels = ['Total Aligned', 'Total Noisy', 'Total UnAligned']

    plot_data(total_aligned, total_noisy, total_unAligned, labels=labels)
    #or
    data_list = [70, 20, 10]
    labels = ['Total Aligned', 'Total Noisy', 'Total UnAligned']

    plot_data(data_list, labels=labels)'''

if __name__ == '__main__':
    #### example1 : 可视化点云和箭头的测试
    # Define start point and direction vector
    start_point = np.array([0, 0, 0])
    direction = np.array([1, 1, 0])  # Example direction vector
    points = np.random.rand(100, 3)  # 100 random points
    display_pts_with_arrow(points, start_point, direction)