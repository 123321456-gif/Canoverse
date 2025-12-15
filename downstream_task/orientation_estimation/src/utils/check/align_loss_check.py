"""
objaverse 的 align data structure

objaverse_alignData:
    temp:
        orgPtsIns : 原始的点云
        OrgSemIns : 原始的semantic dir
    obj:
        orgPtsIns : 原始的点云
        OrgSemIns : 原始的semantic dir

        candidate:
            canPoses : 多种候选的旋转矩阵
            canPtsInses : 多种候选的点云ins
            canSemInses : 多种候选的semantic dir ins
            cds : 与temp的cd
            semantic norm : 与temp的semantic同方向的norm

        align:
            alignPose : 最终的旋转矩阵
            cd : 与temp的cd
            semantic norm(和bbox平面垂直的norm最大的) : 与temp的semantic同方向的norm
            score : 最终的得分
            canPtsIns : 最终的点云ins
            canSemIns : 最终的semantic dir ins
"""

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List
import copy
import matplotlib.pyplot as plt

from src.structure.point_cloud import PointCloud
from src.structure.ins_structure import DataStructure
from src.utils.math import generate_uniform_rots,batch_chamfer_distance, project_vectors
from src.utils.vis import display_two_point_clouds, plot_2d_line, visualize_heatmap,display_pts_with_arrow

# basic data structure
@dataclass
class Org:
    cate_name: str = ''
    ins_name: str = ''
    orgPtsIns: PointCloud = None
    orgSemIns: PointCloud = None

    def rotate(self, rot:torch.Tensor):
        self.orgPtsIns.rotate(rot)
        self.orgSemIns.rotate(rot)

    def vis(self):
        for dir in self.orgSemIns.points:
            print('dir:', dir)
            display_pts_with_arrow(self.orgPtsIns.points.cpu().numpy(), np.array([0,0,0]), dir.cpu().numpy())

@dataclass
class Candidate:
    canPoses: torch.Tensor = torch.eye(3).unsqueeze(0)
    canPtsInses: List[PointCloud] = field(default_factory=lambda: [None])
    canSemInses: List[PointCloud] = field(default_factory=lambda: [None])
    cds: torch.Tensor = torch.tensor(0.0)
    batch_xzNorms: torch.Tensor = torch.tensor(0.0)  # m*n*2
    semConfs: torch.Tensor = torch.tensor(0.0)  # 跟temp 方向同否，以及norm长正相关

@dataclass
class Align:
    alignPose: torch.Tensor = torch.eye(3)
    cd: torch.Tensor = torch.tensor(0.0)
    semantic_norm: torch.Tensor = torch.tensor(0.0)
    score: torch.Tensor = torch.tensor(0.0)
    canPtsIns: PointCloud = None
    canSemIns: PointCloud = None

# middle data structure
@dataclass
class Temp:
    org:Org = field(default_factory=Org)
    sem_xzNorm: torch.Tensor = torch.tensor(0.0) # n*3
@dataclass
class Obj:
    org:Org = field(default_factory=Org)
    candidate: Candidate = field(default_factory=Candidate)
    align: Align = field(default_factory=Align)

# final data structure
@dataclass
class Objaverse_alignData:
    temp: Temp = field(default_factory=Temp)
    obj: Obj = field(default_factory=Obj)
    device:str = 'cuda:0'

    @classmethod
    def from_insStructureJson(cls, temp_path:str, obj_path:str, device:str='cuda:0'):
        # temp Ins
        temp = DataStructure.load(temp_path, device=device)
        # temp info
        # Org
        temp_cate = temp.cate_name
        temp_instance = temp.ins_name
        temp_pts = temp.segObjsIns[0].ptsIns.points
        temp_dirs = temp.segObjsIns[0].semDirsIns.points
        temp_OrgIns = Org(cate_name=temp_cate, ins_name=temp_instance, orgPtsIns=PointCloud(points=temp_pts), orgSemIns=PointCloud(points=temp_dirs))
        # 计算semDirs在x轴和z轴的投影矩阵
        pro_x = project_vectors(temp_dirs, axis='x')
        pro_z = project_vectors(temp_dirs, axis='z')
        temp_sem_xzNorm = torch.stack([pro_x, pro_z], dim=1)

        # obj Ins
        obj = DataStructure.load(obj_path, device=device)   

        # obj info
        # Org
        obj_cate = obj.cate_name
        obj_instance = obj.ins_name
        obj_pts = obj.segObjsIns[0].ptsIns.points
        obj_dirs = obj.segObjsIns[0].semDirsIns.points
        obj_OrgIns = Org(cate_name=obj_cate, ins_name=obj_instance, orgPtsIns=PointCloud(points=obj_pts), orgSemIns=PointCloud(points=obj_dirs))

        # Candidate （都是需要生成的了）
        #1. 生成canPoses
        # Align （都是需要生成的了）
        return cls(temp=Temp(org=temp_OrgIns,sem_xzNorm=temp_sem_xzNorm), obj=Obj(org=obj_OrgIns), device=device)

    def align(self):
        self._Cal_Candidate()
        self._alignment()

    def _Cal_Candidate(self, algleStep:int=5):
        # 1. 生成canPoses
        uniform_rots = generate_uniform_rots('y', algleStep, device=self.device)
        self.obj.candidate.canPoses = uniform_rots
        # 2. 生成canPtsInses 和 canSemInses
        canPts = []
        canSems = []
        for rot in uniform_rots:
            roted_orgIns = copy.deepcopy(self.obj.org)
            roted_orgIns.rotate(rot)
            canPts.append(roted_orgIns.orgPtsIns)
            canSems.append(roted_orgIns.orgSemIns)
        self.obj.candidate.canPtsInses = canPts
        self.obj.candidate.canSemInses = canSems
        # 3. 计算和temp的cd
        batch_canPts = torch.stack([canptIns.points for canptIns in canPts])
        orgPtsIns_points = torch.stack([self.temp.org.orgPtsIns.points for _ in canPts])
        cds = batch_chamfer_distance(batch_canPts, orgPtsIns_points)
        self.obj.candidate.cds = cds

        # 4. 计算和temp的semantic norm
        # 先计算obj的semantic 在x和z轴的投影
        batch_canSems = torch.stack([cansemIns.points for cansemIns in canSems])
        pro_x = project_vectors(batch_canSems, axis='x')
        pro_z = project_vectors(batch_canSems, axis='z')
        batch_xzNorms = torch.stack([pro_x, pro_z], dim=2)
        self.obj.candidate.batch_xzNorms = batch_xzNorms
        # 结合temp的semantic xzNorm， 计算semConfs
        b_temp_xzNorm = self.temp.sem_xzNorm.unsqueeze(0).expand(batch_xzNorms.shape[0], -1, -1)
        semConfs =  b_temp_xzNorm* batch_xzNorms
        self.obj.candidate.semConfs = semConfs



    def _alignment(self):
        # 1. 找出cds中得极小值
        cds = self.obj.candidate.cds
        local_minima_indices = [i for i in range(1, cds.size(0) - 1) if cds[i] < cds[i - 1] and cds[i] < cds[i + 1]]  # 极小值索引
        local_minima_indices = [0] + local_minima_indices + [cds.size(0) - 1]  # 两端的点
        # 2. 找出，local_minima_indices对应得semConfs的值，最大的下标为aligned index
        semConfs = self.obj.candidate.semConfs
        semConf_inLocalMinima = semConfs[local_minima_indices]
        # semConf_inLocalMinima 最大值的索引
        max_index = torch.argmax(semConf_inLocalMinima).item()  # 找到的是扁平化数据的索引
        multi_dim_index = np.unravel_index(max_index, semConf_inLocalMinima.shape)  # 转成多维索引
        self.mainSemDir_index = multi_dim_index[1]
        # 对齐的index
        align_index = local_minima_indices[multi_dim_index[0]]

        # 3. 保存align信息
        self.obj.align.alignPose = self.obj.candidate.canPoses[align_index]
        self.obj.align.cd = cds[align_index]
        self.obj.align.semantic_norm = self.obj.candidate.batch_xzNorms[align_index]
        self.obj.align.score = semConfs[max_index].max()
        self.obj.align.canPtsIns = self.obj.candidate.canPtsInses[align_index]
        self.obj.align.canSemIns = self.obj.candidate.canSemInses[align_index]

        # self.vis(dataType='align')

    def vis(self, dataType='cds'):
        if dataType == 'cds':
            # 用plt画出cds
            y = self.obj.candidate.cds.cpu().numpy()
            plot_2d_line(y)

        if dataType == 'canPts':
            for i, canPts in enumerate(self.obj.candidate.canPtsInses):
                # canPts.vis()
                print('num:', i)
                print('deg:', i*5)
                display_two_point_clouds(self.temp.org.orgPtsIns.points, canPts.points)
                print('----------')

        if dataType == 'align':
            display_two_point_clouds(self.temp.org.orgPtsIns.points, self.obj.align.canPtsIns.points)

        if dataType == 'semConfs':
            # 用plt画出semConfs
            sem_label_index = self.mainSemDir_index #1
            y = self.obj.candidate.semConfs[:, sem_label_index, :].cpu().numpy()  # 
            visualize_heatmap(y)

    # save aligned pts as ply file
    def save(self, save_root:str):
        save_path = os.path.join(save_root, f'{self.obj.org.cate_name}/{self.obj.org.ins_name}_align.ply')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.obj.align.canPtsIns.save2ply(save_path)

if __name__=="__main__":

    device = 'cuda:5'
    # temp
    temp_path = 'results/objaverse/temp/info/airplane_temp_ins_0000.json'
    # new obj
    # obj_path = 'results/objaverse/newObj/info/airplane_newObj_ins_0001.json'
    obj_path = 'results/objaverse//newObj/info/airplane_newObj_ins_0046.json'

    # Objaverse_alignData初始化
    alignDataIns = Objaverse_alignData.from_insStructureJson(temp_path, obj_path, device=device)
    alignDataIns._Cal_Candidate()
    alignDataIns._alignment()
    # alignDataIns.vis()
    # alignDataIns.vis('semConfs')
    # alignDataIns.vis('align')
    # alignDataIns.vis('canPts')
    
    print('alignPose:', alignDataIns.obj.align.alignPose)
    print('alignIndex:', torch.where(alignDataIns.obj.candidate.cds == alignDataIns.obj.align.cd))
    
    # # 感觉像分割问题的check方法 
    alignDataIns.obj.org.vis()

    # #保存对齐点云
    # alignDataIns.save('results/objaverse/align')
    



