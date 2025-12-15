from scipy.spatial.transform import Rotation as R
import numpy as np
import math


def calculate_connacial_rotation_ry(rotation: np.ndarray):
    
    theta_x = rotation[0, 0] + rotation[2, 2]
    theta_y = rotation[0, 2] - rotation[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                        [0.0,            1.0,  0.0           ],
                        [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    
    rotation = rotation @ s_map
    
    return rotation
    
def calculate_connacial_rotation_rz(rotation: np.ndarray):
    
    theta_x = rotation[0, 0] + rotation[1, 1]
    theta_y = rotation[1, 0] - rotation[0, 1]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, -theta_y/r_norm, 0.0],
                        [theta_y/r_norm,  theta_x/r_norm, 0.0],
                        [0.0,             0.0,            1.0]])
    
    rotation = rotation @ s_map.T  
    
    return rotation

def calculate_connacial_rotation_rx(rotation:np.ndarray):

    theta_y = rotation[1, 1] + rotation[2, 2]
    theta_z = rotation[2, 1] - rotation[1, 2]
    r_norm = math.sqrt(theta_y**2 + theta_z**2)
    s_map = np.array([[1.0, 0.0, 0.0],
                      [0.0, theta_y/r_norm, -theta_z/r_norm],
                      [0.0, theta_z/r_norm,  theta_y/r_norm]])

    rotation =  rotation @ s_map.T 

    return rotation

def sphere_sym(rotation: np.ndarray):
    
    return np.eye(3)

def y_180_sym(rotation: np.ndarray):


    y_180 = R.from_euler('zyx', [0, 180, 0], degrees=True).as_matrix()
    
    original_distance = np.linalg.norm(rotation - np.eye(3))
    
    rotated_matrix = rotation @ y_180
    rotated_distance = np.linalg.norm(rotated_matrix - np.eye(3))
    
    if original_distance <= rotated_distance:
        return rotation
    else:
        return rotated_matrix
    


def x_180_sym(rotation: np.ndarray):
    x_180 = R.from_euler('zyx', [0, 0, 180], degrees=True).as_matrix()
    
    original_distance = np.linalg.norm(rotation - np.eye(3))
    
    rotated_matrix = rotation @ x_180
    rotated_distance = np.linalg.norm(rotated_matrix - np.eye(3))
    
    if original_distance <= rotated_distance:
        return rotation
    else:
        return rotated_matrix

def z_180_sym(rotation: np.ndarray):
    z_180 = R.from_euler('zyx', [180, 0, 0], degrees=True).as_matrix()
    
    original_distance = np.linalg.norm(rotation - np.eye(3))
    
    rotated_matrix = rotation @ z_180
    rotated_distance = np.linalg.norm(rotated_matrix - np.eye(3))
    
    if original_distance <= rotated_distance:
        return rotation
    else:
        return rotated_matrix

def y_90_sym(rotation: np.ndarray):

    candidates = [rotation]  
    distances = [np.linalg.norm(rotation - np.eye(3))]
    
    for angle in [90, 180, 270]:
        y_rot = R.from_euler('zyx', [0, angle, 0], degrees=True).as_matrix()
        rotated_matrix = rotation @ y_rot
        candidates.append(rotated_matrix)
        distances.append(np.linalg.norm(rotated_matrix - np.eye(3)))
    
    min_idx = np.argmin(distances)
    return candidates[min_idx]

def x_90_sym(rotation: np.ndarray):
    
    candidates = [rotation] 
    distances = [np.linalg.norm(rotation - np.eye(3))]
    
    for angle in [90, 180, 270]:
        x_rot = R.from_euler('zyx', [0, 0, angle], degrees=True).as_matrix()
        rotated_matrix = rotation @ x_rot
        candidates.append(rotated_matrix)
        distances.append(np.linalg.norm(rotated_matrix - np.eye(3)))
    
    min_idx = np.argmin(distances)
    return candidates[min_idx]

def z_90_sym(rotation: np.ndarray):

    candidates = [rotation]  
    distances = [np.linalg.norm(rotation - np.eye(3))]
    
    for angle in [90, 180, 270]:
        z_rot = R.from_euler('zyx', [angle, 0, 0], degrees=True).as_matrix()
        rotated_matrix = rotation @ z_rot
        candidates.append(rotated_matrix)
        distances.append(np.linalg.norm(rotated_matrix - np.eye(3)))
    
    min_idx = np.argmin(distances)
    return candidates[min_idx]

def sym_90_any_axis(rotation: np.ndarray):
    
    candidates = [rotation]  
    distances = [np.linalg.norm(rotation - np.eye(3))]
    
    for angle in [90, 180, 270]:
        for axis in ['x', 'y', 'z']:
            if axis == 'x':
                rot = R.from_euler('zyx', [0, 0, angle], degrees=True).as_matrix()
            elif axis == 'y':
                rot = R.from_euler('zyx', [0, angle, 0], degrees=True).as_matrix()
            else:
                rot = R.from_euler('zyx', [angle, 0, 0], degrees=True).as_matrix()
            
            rotated_matrix = rotation @ rot
            candidates.append(rotated_matrix)
            distances.append(np.linalg.norm(rotated_matrix - np.eye(3)))
    
    min_idx = np.argmin(distances)
    return candidates[min_idx]


if __name__ == "__main__":
    
    pass
    
    