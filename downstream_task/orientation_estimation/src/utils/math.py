import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon

# 1. 旋转矩阵转欧拉角
def rotation_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    将一个旋转矩阵转换为欧拉角（XYZ顺序）
    :param rotation_matrix: 3x3的旋转矩阵
    :return: 对应的欧拉角（弧度）
    """
    rotation = R.from_matrix(rotation_matrix)
    euler = rotation.as_euler('xyz', degrees=False)
    return euler
def batch_rotation_to_euler(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    批量将旋转矩阵转换为欧拉角
    :param rotation_matrices: 形状为 (N, 3, 3) 的旋转矩阵数组
    :return: 形状为 (N, 3) 的欧拉角数组
    """
    eulers = np.array([rotation_to_euler(rot_matrix) for rot_matrix in rotation_matrices])
    return eulers

# 2. 欧拉角转旋转矩阵
def euler_to_rotation(euler_angles):
    """
    将欧拉角（XYZ顺序）转换为旋转矩阵
    :param euler_angles: 欧拉角数组（弧度），形状为 (3,)
    :return: 对应的旋转矩阵，形状为 (3, 3)
    """
    rotation = R.from_euler('xyz', euler_angles, degrees=False) # False
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix
def batch_euler_to_rotation(euler_angles_array):
    """
    批量将欧拉角转换为旋转矩阵
    :param euler_angles_array: 形状为 (N, 3) 的欧拉角数组
    :return: 形状为 (N, 3, 3) 的旋转矩阵数组
    """
    rotation_matrices = np.array([euler_to_rotation(euler_angles) for euler_angles in euler_angles_array])
    return rotation_matrices


# 旋转轴旋转角转旋转矩阵
def axisAngle_to_rotation(axis:np.array, angle:float)->np.array:
    """
    通过旋转轴和旋转角度生成旋转矩阵
    :param axis: 旋转轴
    :param angle: 旋转角度
    :return: 旋转矩阵
    """
    rotation = R.from_rotvec(angle * axis)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


# 3D空间中，输入多边形的顶点， 和平面外的一个点C
# 计算 多边形为地面，C在Y轴正半轴时，对应的旋转矩阵
def Rot_plane2grounding(polygons, centor:np.ndarray):

    """计算旋转矩阵R，使得三角面片与XOZ平行且点云都在面片上方"""
    # 提取面片的顶点
    p1, p2, p3 = polygons[:3, :]
    triangle_vertexs = np.array([p1, p2, p3])
    # 计算面片的法向量
    normal = calculate_normal(p1, p2, p3)
    # 目标法向量，我们希望面片与XOZ平行，因此目标法向量为Y轴正方向
    target_normal = np.array([0, 1, 0])
    # 计算旋转矩阵
    R = rotation_matrix_from_vectors(normal, target_normal)

    # 对应的旋转平面的顶点
    rotated_triangle_vertexs = triangle_vertexs @ R.T

    # 平面到质点的向量，是否和Y正同向
    center_v = centor-rotated_triangle_vertexs[0]
    y_v = np.array([0,1,0])
    dot_product = np.dot(center_v, y_v)

    # print('dot_product: ', dot_product)
    # asdf
    if dot_product < 0: 
        # 如果有点在XOZ平面下方，绕X轴旋转180度
        R_flip = np.eye(3)
        R_flip[1,1] = -1
        R_flip[2,2] = -1
        R = R_flip @ R
    return R

# 给定3D 平面上的三个点，计算平面法向量
def calculate_normal(p1, p2, p3):
    """计算通过三点定义的面片的法向量"""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)
# check normal is right
def _check_normal(face, normal:np.ndarray):
    # check it's right
    side_vector1 = face[1] - face[0]
    side_vector2 = face[2] - face[0]
    side_vector3 = face[2] - face[1]
    # 判断adjusted_normal 是否和side_vector1, side_vector2, side_vector3 垂直
    print(np.dot(normal, side_vector1) )
    print(np.dot(normal, side_vector2) )
    print(np.dot(normal, side_vector3) )
    # return True

# 3D中 输入两个向量，计算从vec1转到vec2的旋转矩阵
def rotation_matrix_from_vectors(vec1:np.array, vec2:np.array) -> np.array:
    """根据两个向量计算旋转矩阵，使vec1旋转到vec2的方向"""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # 先排除两个向量平行的情况
    rotation_angle = np.arccos(np.dot(a, b))  

    if rotation_angle == 0 :
        rotation_matrix = np.eye(3)
        return rotation_matrix
    elif rotation_angle>3.141 :
        rotation_matrix = np.eye(3)
        rotation_matrix[1,1] = -1.0
        rotation_matrix[2,2] = -1.0
        return rotation_matrix
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

# 给定旋转轴，和两个向量，计算旋转矩阵，使得vec1旋转到vec2的方向
def compute_rotation_angle_and_matrix(axis:torch.tensor, v1:torch.tensor, v2:torch.tensor)-> np.array:  
    """
    计算旋转角度和旋转矩阵，使得v1通过旋转轴旋转后与v2在旋转轴垂直的平面上对齐
    
    :param axis: 旋转轴 (单位向量) - Tensor，形状 [3]
    :param v1: 向量 v1 - Tensor，形状 [3]
    :param v2: 向量 v2 - Tensor，形状 [3]
    :return: 旋转角度，旋转矩阵
    """
    # 将 v1 和 v2 投影到与旋转轴垂直的平面
    axis = axis / axis.norm()  # 确保旋转轴是单位向量
    v1_parallel = v1 - torch.dot(v1, axis) * axis
    v2_parallel = v2 - torch.dot(v2, axis) * axis
    
    # 计算 v1 和 v2 在平面上的夹角
    cos_angle = torch.dot(v1_parallel, v2_parallel) / (v1_parallel.norm() * v2_parallel.norm())
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))  # 限制在 -1 到 1 之间，避免浮动误差

    # 使用叉积判断旋转方向
    cross_product = torch.cross(v1_parallel, v2_parallel)
    if torch.dot(cross_product, axis) < 0:
        angle = -angle  # 如果方向相反，则调整角度的符号
    
    # 计算旋转轴和旋转角度对应的旋转矩阵
    rotation_mat = axisAngle_to_rotation(axis.numpy(), angle)
    
    return angle, rotation_mat

# 计算3D空间中，多边形的面积
def calculate_area(faces:list[list[float]]) -> float:
    """计算3D空间中多边形的面积
    输入：多边形的顶点坐标 list[list[float]]
    步骤：
        1. 先将3D多边形转成2D多边形
            a. 计算法向量
            b. 旋转多边形和坐标平面平行
            c. 取出对应的轴坐标, 即使2D多边形
        2. 重排列多边形的顶点（以正方形为例，乱序无法计算）
        3. 计算多边形的面积
    # 
    """
    # 1. 计算法向量 并 转成2D多边形
    faces_np = np.array(faces)
    rot = Rot_plane2grounding(faces_np, np.array([0, 0, 0]))
    poly_verts_xoz = (faces_np @ rot.T)[:,[0,2]]
    # 2. 重排列多边形的顶点
    def sort_points(points):
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]
    sorted_projected_face = sort_points(poly_verts_xoz)
    # 3. 计算多边形的面积
    polygon = Polygon(sorted_projected_face)
    area = polygon.area
    return area


# 计算点云中最远两个点的距离
def farthest_points_distance(points):
    """
    计算点云中最远的两个点之间的距离
    :param points: 输入的点云数据，形状为 (N, 3) 的 Tensor，其中 N 为点的数量
    :return: 最远两个点之间的距离
    """
    # 计算点对之间的距离
    distances = torch.cdist(points, points)  # 计算点云中所有点对的欧几里得距离
    
    # 找到最大距离
    max_distance = torch.max(distances)  # 最大距离
    
    return max_distance.item()  # 返回最大距离的数值

# 2D交并比的计算
# https://blog.csdn.net/hxxjxw/article/details/119719824
def IOU(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou
# 3D IOU的计算
def calculate_3d_iou(bbox1:torch.tensor, bbox2:torch.tensor):
    # 计算交集的bbox
    inter_min = torch.max(bbox1[0], bbox2[0])
    inter_max = torch.min(bbox1[1], bbox2[1])
    inter_bbox = torch.stack([inter_min, inter_max])

    # 计算交集的体积
    inter_volume = torch.prod(inter_bbox[1] - inter_bbox[0])

    # 计算并集的体积
    volume1 = torch.prod(bbox1[1] - bbox1[0])
    volume2 = torch.prod(bbox2[1] - bbox2[0])
    union_volume = volume1 + volume2 - inter_volume

    # 计算3D iou
    iou = inter_volume / union_volume
    return iou


# generate random rot
def generate_random_rot()->torch.tensor:
    r = R.random()
    return torch.tensor(r.as_matrix(), dtype=torch.float32)

# 将3D空间中的向量分解为水平方向和竖直方向
def decompose_vector(dir:torch.tensor)->torch.tensor:
    parallel_y = torch.tensor([0.0, dir[1], 0.0], device=dir.device, dtype=torch.float32)
    perpendicular_y = dir - parallel_y
    return parallel_y, perpendicular_y

# 3维空间中，平行于xoz平面的两个向量A，B ； 将A向量通过旋转矩阵R，变换到B向量
def rotate_plane_vector(sor_vec:torch.tensor, tar_vec:torch.tensor)->torch.tensor:
    # Ensure vectors are normalized
    temp_perpendicular_y_normalized = tar_vec / torch.norm(tar_vec)
    obj_perpendicular_y_normalized = sor_vec / torch.norm(sor_vec)

    # Compute the angle between the vectors
    dot_product = torch.dot(temp_perpendicular_y_normalized, obj_perpendicular_y_normalized)
    angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # Clamp for numerical stability

    # Check direction of rotation (cross product y component sign)
    cross_product = torch.cross(obj_perpendicular_y_normalized, temp_perpendicular_y_normalized)
    direction = torch.sign(cross_product[1])

    # Rotation matrix around Y-axis
    R = torch.tensor([
        [torch.cos(direction * angle), 0, torch.sin(direction * angle)],
        [0, 1, 0],
        [-torch.sin(direction * angle), 0, torch.cos(direction * angle)]
    ], device=sor_vec.device, dtype=torch.float32)
    return R

# 计算3D空间中两个向量的夹角
def angle_between_vectors(v1, v2):
    # Ensure the vectors are float tensors
    v1 = v1.float()
    v2 = v2.float()
    
    # Compute the dot product
    dot_product = torch.dot(v1, v2)
    
    # Compute the norms (magnitudes) of the vectors
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)
    
    # Compute the cosine of the angle
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    
    # Compute the angle in radians
    angle_rad = torch.acos(cosine_angle)
    
    # Convert the angle to degrees
    angle_deg = angle_rad * (180.0 / math.pi)
    
    return angle_rad.item(), angle_deg.item()

# CD loss函数
def batch_chamfer_distance(point_cloud1:torch.tensor, point_cloud2:torch.tensor):
    B, N, _ = point_cloud1.shape
    _, M, _ = point_cloud2.shape

    point_cloud1_expanded = point_cloud1.unsqueeze(2)
    point_cloud2_expanded = point_cloud2.unsqueeze(1)

    distances = torch.sum((point_cloud1_expanded - point_cloud2_expanded) ** 2, dim=-1)

    min_dist_point_cloud1 = torch.min(distances, dim=2)[0]
    min_dist_point_cloud2 = torch.min(distances, dim=1)[0]

    chamfer_distance = torch.mean(min_dist_point_cloud1, dim=1) + torch.mean(min_dist_point_cloud2, dim=1)

    return chamfer_distance

# 生成3D的rots，均匀的在一个2D平面空间
# 如: 生成的rots，都是绕Y轴旋转的
def generate_uniform_rots(around_axis:str, angleStep:int, device:str='cpu')->torch.tensor:
    if around_axis == 'y':
        rots = []
        angles = []
        # 沿Y轴旋转360度，每10度计算一次距离
        for angle in range(0, 360, angleStep):
            angle_radians = np.deg2rad(angle)
            rotation_matrix = torch.tensor([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]], dtype=torch.float32)
            rots.append(rotation_matrix)
            angles.append(angle)

        return torch.stack(rots).to(device)
    else:
        raise ValueError('Only support rotation around y axis')


# 计算3D向量在坐标轴上的投影（和正方向相同为正，否则为负）
def project_vectors(tensor:torch.tensor, axis) -> torch.tensor:
    if len(tensor.shape) ==2:  # 无batch的处理
        assert tensor.shape[1] == 3 #输入张量的形状应该是 (n, 3)
        # 根据指定的坐标轴选择相应的列
        if axis == 'x':
            projection = tensor[:, 0]
        elif axis == 'y':
            projection = tensor[:, 1]
        elif axis == 'z':
            projection = tensor[:, 2]
        else:
            raise ValueError('Axis must be one of x, y, or z')
    elif len(tensor.shape) == 3:  # 有batch的处理
        assert tensor.shape[2] == 3
        if axis == 'x':
            projection = tensor[:, :, 0]
        elif axis == 'y':
            projection = tensor[:, :, 1]
        elif axis == 'z':
            projection = tensor[:, :, 2]
        else:
            raise ValueError('Axis must be one of x, y, or z')

    return projection

# 根据旋转矩阵，计算变换前后坐标系各轴的夹角
def calculate_axis_angles(rotation_matrix:torch.tensor):
    # 确保输入是一个 3x3 矩阵
    if rotation_matrix.shape != (3, 3):
        raise ValueError("The input rotation matrix must be 3x3.")

    device = rotation_matrix.device
    # 定义标准基向量
    x_axis = torch.tensor([1.0, 0.0, 0.0]).to(device)
    y_axis = torch.tensor([0.0, 1.0, 0.0]).to(device)
    z_axis = torch.tensor([0.0, 0.0, 1.0]).to(device)

    # 计算变换后的基向量
    x_axis_transformed = rotation_matrix @ x_axis
    y_axis_transformed = rotation_matrix @ y_axis
    z_axis_transformed = rotation_matrix @ z_axis

    # 计算夹角的函数
    def angle_between_vectors(v1, v2):
        cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        angle = torch.acos(cos_theta)
        return math.degrees(angle)

    # 计算变换前后基向量的夹角
    angle_x = angle_between_vectors(x_axis, x_axis_transformed)
    angle_y = angle_between_vectors(y_axis, y_axis_transformed)
    angle_z = angle_between_vectors(z_axis, z_axis_transformed)

    return angle_x, angle_y, angle_z


# 计算两个旋转矩阵的误差（距离）
def ErrNdeg(rPred, rGt):
    """
    计算两个旋转矩阵之间的旋转误差。

    参数：
        rPred : numpy数组, 3x3的旋转矩阵，预测的旋转矩阵
        rGt   : numpy数组, 3x3的旋转矩阵，地面真实的旋转矩阵

    返回：
        errRot: 旋转误差（以度为单位）
    """
    
    # 计算旋转误差
    tt = np.transpose(rPred)
    tmpR = np.dot(tt, rGt)
    trace = tmpR[0][0] + tmpR[1][1] + tmpR[2][2]

    # 确保trace值在合法范围内，避免超出 [-1, 3] 的情况
    if trace > 3:
        trace = 3
    elif trace < -1:
        trace = -1

    # 计算旋转误差，单位为度
    errRot = math.acos((trace - 1) / 2) * 180 / math.pi

    return errRot

