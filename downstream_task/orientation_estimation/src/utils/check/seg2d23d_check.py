import torch
import numpy as np
import os
from pytorch3d.structures import Meshes
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

from src.structure.pytorch_mesh import PytorchMesh
from src.seg3d.seg3d_utils import mask_normalization, semantic_2D_to_3d, cluster_confidences_tensor
from src.utils.h5_utils import load_masks_hdf5
from src.utils.pytorch_utils import find_pix_to_face, add_texture_to_mesh
from src.utils.vis import vis_pytorch3d_mesh, get_colors, plot_2d_line
from src.structure.point_cloud import PointCloud
from src.seg3d.seg3d import calculate_weighted_direction_vectors_tensor


"""project 2D semantics segmentation to 3D space

input:
    2D semantics segmentation masks
    3D mesh in support pose (corresponding to the 2D segmentation)
    the number of semantic labels

Returns:
    3D semantics segmentation
"""

# # 将语义转为方向向量，tensor版
# def calculate_weighted_direction_vectors_tensor(semantics, verts):
#     semantics = torch.tensor(semantics, dtype=torch.float32)
#     verts = torch.tensor(verts, dtype=torch.float32)
#     total_weights = torch.sum(semantics, dim=0)
#     weighted_sum = torch.matmul(semantics.T, verts)  
#     total_weights = total_weights
#     valid_weights = total_weights > 0  
#     weights = total_weights.unsqueeze(-1)  
#     weighted_centers = torch.zeros_like(weighted_sum)
#     weighted_centers[valid_weights] = weighted_sum[valid_weights] / weights[valid_weights]
#     return weighted_centers

# 将2D包围盒画在2D图像上
def draw_bbox_on_img(bbox, img_path):
    img = cv2.imread(img_path)
    bbox = [int(coord) for coord in bbox]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.title('Image with bbox')
    plt.axis('off')  # Hide axes
    plt.show()

    # return img


class seg3d:  
    def __init__(self, seg2D_path:str, mesh:Meshes, part_labs:list, device, vis=None, debug=None):

        
        self.seg2D_path = seg2D_path  # masks.h5
        self.mesh = mesh  
        self.parts_num = len(part_labs)
        self.device = device

        self._data_loading()
        #print('data loading finish!')
        self._project_2d_to_3d()

        # if debug is not None:
        #     self.process_single_object(debug)
        # else:
        #     self.process_data()
        
        # if vis is not None:
        #     self.visualize()

    def _data_loading(self):
        # 3D mesh
        self.support_mesh = self.mesh.clone()
        # 2D mask
        self.masks = load_masks_hdf5(self.seg2D_path)

    def _project_2d_to_3d(self):
        # 开始遍历全部的图片
        muti_views_verts_colors = torch.zeros(len(self.support_mesh.verts_packed()), self.parts_num, dtype=torch.float32, device=self.device)
        for view_idx in range(10):
            print('view_idx:', view_idx)
            # 3. 找到2D和3D的对应关系
            # 打印mesh面片得个数
            pix_to_face, depth = find_pix_to_face(self.support_mesh, view_idx, self.device)

            # # 画出深度图，看跟渲染的是否相同
            # plt.imshow(depth[0].cpu().numpy())
            # plt.show()
            
            # 4. 将2D投影到3D上
            maski = self.masks[view_idx] 

            # # check
            # # print(maski)
            # for mask_tuple in maski:
            #     mask, label, mask_bbox = mask_tuple
            #     print(mask_bbox)
            
            #     draw_bbox_on_img(mask_bbox, 'results/objaverse/newObj/results/airplane/ins_0010/rot_0/rendered_img/' + str(view_idx) + '.png')
            # print('----------------')
            # # asdf
            
            if len(maski) == 0:  # 过滤掉没有mask的情况
                continue
            obj_bbox = self._find_obj_bbox(depth)
            if obj_bbox is None: # 图片上什么都没有
                continue
            maski = mask_normalization(maski, obj_bbox) # 对mask进行清洗

            if len(maski) == 0:  # 过滤掉没有mask的情况
                continue

            verts_colors = semantic_2D_to_3d(maski, pix_to_face, self.support_mesh, self.parts_num, self.device)  # 投到3D上

            # # check
            # print(verts_colors.shape)
            # # 看verts_colors的值域
            # print(torch.max(verts_colors, dim=0)[0])
            # print(torch.min(verts_colors, dim=0)[0])
            # asdf

            # 累加
            muti_views_verts_colors += verts_colors
            self.muti_views_verts_colors = muti_views_verts_colors
        
        # 全局变量
        self.muti_views_verts_colors = muti_views_verts_colors

    # 找到obj在图像上的2D包围盒
    def _find_obj_bbox(self, depth):
        mask = depth != -1
        coords = torch.where(mask[0, :, :, 0])
        if len(coords[0]) == 0:
            return None
        bbox = [coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()]
        bbox = [int(i) for i in bbox]
        return bbox

    def visualize(self, muti_views_verts_colors=None):
        
        # 可视化是外输入得矩阵，还是内部得矩阵
        if muti_views_verts_colors is None:
            muti_views_verts_colors = self.muti_views_verts_colors
        else:
            muti_views_verts_colors = muti_views_verts_colors

        # 5. 可视化check
        normalized_weight = torch.max(muti_views_verts_colors, dim=0)[0]
        normalized_weight[normalized_weight == 0] = 100
        normalized_verts_colors = muti_views_verts_colors / normalized_weight # muti_views_verts_colors.sum(dim=1, keepdim=True)

        # normalized_verts_colors中第一列大于0.5的索引， semantic_color_mesh对应索引那一行的第0个给1
        semantic_color_mesh = torch.zeros_like(normalized_verts_colors, device=self.device)
        for lie in range(self.parts_num):
            semantic_confi = cluster_confidences_tensor(normalized_verts_colors[:,lie])
            semantic_color_mesh = get_colors(semantic_confi, cmap='RdBu')
            semantic_color_mesh = torch.tensor(semantic_color_mesh, dtype=torch.float32, device=self.device)

            print(normalized_verts_colors[:,lie].shape, semantic_color_mesh.shape)
        
            vis_mesh = add_texture_to_mesh(self.support_mesh, semantic_color_mesh)
            vis_pytorch3d_mesh(vis_mesh)

    def save_results(self, save_path=None):
        muti_views_verts_colors = self.muti_views_verts_colors

        # 有些语义label 一个都没有 torch.max(muti_views_verts_colors, dim=0)[0] 就会有0
        # 将torch.max(muti_views_verts_colors, dim=0)[0] 为0的地方，全部置为100
        normalized_weight = torch.max(muti_views_verts_colors, dim=0)[0]
        normalized_weight[normalized_weight == 0] = 100
        
        # 5. 可视化check
        normalized_verts_colors = muti_views_verts_colors / normalized_weight # muti_views_verts_colors.sum(dim=1, keepdim=True)

        semantic_color_mesh = torch.zeros_like(normalized_verts_colors, device=self.device)
        parts_semantic_confis = torch.zeros_like(normalized_verts_colors, device=self.device)
        for lie in range(self.parts_num):


            semantic_confi = cluster_confidences_tensor(normalized_verts_colors[:,lie])

            # 异常处理
            if normalized_weight[lie] == 100:
                semantic_confi = torch.zeros_like(semantic_confi) #+0.0001

            parts_semantic_confis[:, lie] = semantic_confi

            semantic_color_mesh = get_colors(semantic_confi, cmap='RdBu')
            semantic_color_mesh = torch.tensor(semantic_color_mesh, dtype=torch.float32, device=self.device)
            vis_mesh = add_texture_to_mesh(self.support_mesh, semantic_color_mesh)
            # vis_pytorch3d_mesh(vis_mesh)
        # # 对概率得结果进行聚类，然后保存结果
        # normalized_verts_colors = self.muti_views_verts_colors / torch.max(self.muti_views_verts_colors, dim=0)[0] # muti_views_verts_colors.sum(dim=1, keepdim=True)
        # # 1. 遍历self.muti_views_verts_colors得每一列，进行聚类
        # parts_semantic_confis = torch.zeros_like(normalized_verts_colors, device=self.device)
        # for i in range(self.parts_num):
        #     confidences = self.muti_views_verts_colors[:,i]
        #     cluster_confi = cluster_confidences_tensor(confidences)
        #     parts_semantic_confis[:,i] = cluster_confi


        #     semantic_color_mesh = get_colors(cluster_confi, cmap='RdBu')
        #     semantic_color_mesh = torch.tensor(semantic_color_mesh, dtype=torch.float32, device=self.device)
        #     print(normalized_verts_colors[:,i].shape, semantic_color_mesh.shape)
        #     vis_mesh = add_texture_to_mesh(self.support_mesh, semantic_color_mesh)
        #     vis_pytorch3d_mesh(vis_mesh)
            
        # 2. 保存为npy
        if save_path is None:
            self.seg2D_path.split('glip_pred/')  # 例，results/0417-DREDS/aeroplane/results/0_random0/rot0/glip_pred/masks.h5
            # 分出来save path
            save_path = self.seg2D_path.split('glip_pred/')[0] + 'semantic3d/semantics.npy'
            os.makedirs(save_path.split('semantics.npy')[0], exist_ok=True)
            #print('save_path:', save_path)
        np.save(save_path, parts_semantic_confis.cpu().numpy())
        # 把verts 和 faces 单独存为npy
        verts_path = save_path.split('semantics.npy')[0] + 'support_verts.npy'
        faces_path = save_path.split('semantics.npy')[0] + 'support_faces.npy'
        np.save(verts_path, self.support_mesh.verts_padded()[0].cpu().numpy())
        np.save(faces_path, self.support_mesh.faces_padded()[0].cpu().numpy())
        # # 保存对应得self.support_mesh
        # save_mesh_path = self.seg2D_path.split('glip_pred/')[0] + 'semantic3d/support_mesh.obj'
        # save_obj(save_mesh_path, self.support_mesh.verts_padded()[0], self.support_mesh.faces_padded()[0])
        return save_path
    

    def _obtain_seg3d_dir(self, save_path) -> PointCloud:
        save_dir = os.path.dirname(save_path)
        vets = np.load(os.path.join(save_dir, 'support_verts.npy'))
        sem_labs = np.load(save_path)

        sem_dirs = calculate_weighted_direction_vectors_tensor(sem_labs, vets)
        print('sem_dirs:', sem_dirs)
        return PointCloud(sem_dirs) # PointCloud(torch.tensor(sem_dirs))

if __name__=="__main__":

    # fume_hood ins_0016
    masks_path = 'results/objaverse_lvis/newObj/results/kitchen_table/ins_0019/rot_0/glip_pred/masks.h5'
    mesh_dir = "data/interim/objaverse_lvis_dataset/newObj/kitchen_table/ins_0019/"
    # 获得路径下的.obj文件
    mesh_path = [mesh_dir + file for file in os.listdir(mesh_dir) if file.endswith('.obj')][0]

    device = torch.device("cuda:0")
    pyMesh_ins = PytorchMesh.read_from_obj(mesh_path, device=device).meshes
    # semantics labels
    # part_labs = ["signal_receiver", "mast", "base"]
    part_labs = ["vent", "fan", "hood", "filter"]
    # part_labs = ['tail']

    # seg 3d
    seg3d_ins = seg3d(masks_path, pyMesh_ins, part_labs, device)
    save_path = seg3d_ins.save_results()
    seg3d_ins._obtain_seg3d_dir(save_path)
    seg3d_ins.visualize()
    # seg3d_ins.save_results()
