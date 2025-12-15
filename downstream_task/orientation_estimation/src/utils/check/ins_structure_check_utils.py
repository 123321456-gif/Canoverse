"""用于测试 ins_structure load 的数据相互之间是否正确
"""

import numpy as np
import torch

from src.structure.ins_structure import DataStructure, ObjBasis
from src.utils.vis import display_two_point_clouds
from src.structure.point_cloud import PointCloud



# check 1: org pts 经过 init pose 和 init pts相同
def Check_init(obj):
    # check 1: org pts 经过 init pose 和 init pts相同
    print(obj.shape.pts)
    orgPts = np.load(obj.shape.pts)
    initPose = obj.initial_pose.cpu().numpy()
    # 变换得到initPts
    initPts = np.matmul(initPose[:3, :3], orgPts.T).T
    # read initPts
    readInitPts = obj.initObjIns.ptsIns.points.cpu().numpy()

    # 可视化两个点云
    display_two_point_clouds(initPts, readInitPts)

# check 2: segPose 是否正确: 是否都是支撑面在地面上
def Check_segPose(obj):
    initPts = obj.initObjIns.ptsIns.points
    for (segObj, segPose) in zip(obj.segObjsIns, obj.segmentation.poses):
        # 读取得pts对不对
        segObj.vis()
        # 读取得poses对不对
        tranSegIns = ObjBasis(ptsIns=PointCloud(initPts))
        tranSegIns.rotate(segPose)
        tranSegIns.vis()
        print('-------------------')

# check 3: 3D semantic segmentation是否正确
def Check_semantic(obj):
    for segObjIns in obj.segObjsIns:
        segObjIns.vis(vis_type='sem')
        print(segObjIns.semDirsIns.points)
        # 计算每个向量的模长
        magnitudes = torch.norm(segObjIns.semDirsIns.points, dim=1)
        print('norm:', magnitudes)

if __name__=='__main__':
    # json path
    # path = 'results/camera/temp/info/camera_temp_ins_0000.json'
    # path = 'results/camera/newObj/info/camera_newObj_ins_0005.json'
    path = 'results/objaverse/newObj/info/airplane_newObj_ins_0001.json'
    device = 'cuda:5'
    obj = DataStructure.load(path, device=device)

    # check init
    Check_init(obj)

    # check segPose 
    Check_segPose(obj)

    # check semantic
    Check_semantic(obj)

    
