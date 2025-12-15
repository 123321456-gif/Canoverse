import math
import os
import sys
import numpy as np
from pathlib import Path
import open3d as o3d
import logging
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json

from torch.utils.data import Dataset
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from model.PTS_VI_Net import Net
class PTSTestDataset(Dataset):
    def __init__(self,
            config, 
            resolution=64,
            ds_rate=2
    ):
        self.config = config

        self.resolution = resolution
        self.ds_rate = ds_rate
        self.sample_num = self.config.sample_num
        
        self.cate_sym_map = json.load(open('sym_process/cate_sym_label.json'))
        pts_dir_path = config.data_dir
        if os.path.exists(f'{pts_dir_path}_cache'):
            self.npy_paths = json.load(open(f'{pts_dir_path}_cache/cached_npy_paths.json'))
            self.npy_paths = [os.path.join(pts_dir_path, p) for p in self.npy_paths]
        else:
            self.npy_paths = list(tqdm(Path(pts_dir_path).rglob('*.npy')))
            os.makedirs(f'{pts_dir_path}_cache')
            json.dump([str(p.relative_to(pts_dir_path)) for p in self.npy_paths], open(f'{pts_dir_path}_cache/cached_npy_paths.json', 'w'))

        self.dir_classid_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(pts_dir_path)))}
        
        print('{} npys files are found.'.format(len(self.npy_paths)))
        if len(self.npy_paths) == 0:
            raise ValueError('No pts files found in {}'.format(pts_dir_path))
        

    def __len__(self):

        return len(self.npy_paths)
        
    def reset(self):
        pass

    def __getitem__(self, index):
        
            data_dict = self._read_data(index)
            return data_dict
        
    def _read_data(self, index):
        
        npy_nums = len(self.npy_paths)

        npy_path = Path(self.npy_paths[index%npy_nums])

        dir_name = Path(npy_path).parents[1].name
        cat_id = self.dir_classid_map[dir_name]
        
        data = np.load(npy_path)
        points = data[:,:3]
        colors = data[:,3:6]
        normals = data[:,6:9]

        rotation = Rotation.random().as_matrix().astype(np.float32)
        
        points = points / np.linalg.norm(points, axis=-1).max()

        points = points.astype(np.float32) @ rotation.T
        normals = normals.astype(np.float32) @ rotation.T


        translation = np.zeros((3,)).astype(np.float32)

        size = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).astype(np.float32)


        v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
        rho = np.arctan2(v[1], v[0])
        if v[1]<0:
            rho += 2*np.pi
        phi = np.arccos(v[2])

        vp_rotation = np.array([
            [np.cos(rho),-np.sin(rho),0],
            [np.sin(rho), np.cos(rho),0],
            [0,0,1]
        ]) @ np.array([
            [np.cos(phi),0,np.sin(phi)],
            [0,1,0],
            [-np.sin(phi),0,np.cos(phi)],
        ])
        ip_rotation = vp_rotation.T @ rotation

        rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
        phi_label = int(phi/np.pi*(self.resolution//self.ds_rate))

        ret_dict = {}
        ret_dict['rgb'] = torch.FloatTensor(colors)  # N*3
        ret_dict['pts'] = torch.FloatTensor(points)  # N*3
        ret_dict['normals'] = torch.FloatTensor(normals)  # N*3
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        ret_dict['npy_path'] = str(npy_path)

        ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
        ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
        ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
        ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)

        return ret_dict


def ErrNdeg(rPred, rGt):
    tmpR = np.dot(rPred, rGt.T)
    trace = tmpR[0][0] + tmpR[1][1] + tmpR[2][2]
    trace = max(min(trace, 3), -1)
    errRot = math.acos((trace - 1) / 2) * 180 / math.pi
    return errRot

def get_logger(epoch):
    logger = logging.getLogger('pts_test_epoch_' + str(epoch))
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    file_handler = logging.FileHandler(f'log/pts_test_epoch_{epoch}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def cal_err_rot_sym_axis(R_pred, R_gt, sym_type):
        
        R1 = R_pred
        R2 = R_gt
        if sym_type == 'y-180':
            y_180_RT = np.diag([-1.0, 1.0, -1.0])
            R = R1 @ R2.transpose()
            R_rot = R1 @ y_180_RT @ R2.transpose()
            theta = min(np.arccos((np.trace(R) - 1) / 2),
                        np.arccos((np.trace(R_rot) - 1) / 2))
            return theta * 180 / np.pi
        elif sym_type == 'y-symmetric':
            y = np.array([0, 1, 0])
            y1 = R1 @ y
            y2 = R2 @ y
            theta = np.arccos(
                y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
            return theta * 180 / np.pi
        elif sym_type == 'x-180':
            x_180_RT = np.diag([1.0, -1.0, -1.0])
            R = R1 @ R2.transpose()
            R_rot = R1 @ x_180_RT @ R2.transpose()
            theta = min(np.arccos((np.trace(R) - 1) / 2),
                        np.arccos((np.trace(R_rot) - 1) / 2))
            return theta * 180 / np.pi
        elif sym_type == 'x-symmetric':
            x = np.array([1, 0, 0])
            x1 = R1 @ x
            x2 = R2 @ x
            theta = np.arccos(
                x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
            return theta * 180 / np.pi
        elif sym_type == 'z-180':
            z_180_RT = np.diag([-1.0, -1.0, 1.0])
            R = R1 @ R2.transpose()
            R_rot = R1 @ z_180_RT @ R2.transpose()
            theta = min(np.arccos((np.trace(R) - 1) / 2),
                        np.arccos((np.trace(R_rot) - 1) / 2))
            return theta * 180 / np.pi
        elif sym_type == 'z-symmetric':
            z = np.array([0, 0, 1])
            z1 = R1 @ z
            z2 = R2 @ z
            theta = np.arccos(
                z1.dot(z2) / (np.linalg.norm(z1) * np.linalg.norm(z2)))
            return theta * 180 / np.pi
        elif sym_type == '90-symmetric':
            R_candidates = []
            R_candidates.append(R2)
            for angle in [90, 180, 270]:
                y_rot = Rotation.from_euler('zyx', [0, angle, 0], degrees=True).as_matrix()
                R_candidates.append(R2 @ y_rot)
                x_rot = Rotation.from_euler('zyx', [0, 0, angle], degrees=True).as_matrix()
                R_candidates.append(R2 @ x_rot)
                z_rot = Rotation.from_euler('zyx', [angle, 0, 0], degrees=True).as_matrix()
                R_candidates.append(R2 @ z_rot)
            min_err = 180.0
            for R_cand in R_candidates:
                R_diff = R1 @ R_cand.T
                theta = np.arccos((np.trace(R_diff) - 1) / 2)
                err_deg = theta * 180 / np.pi
                if err_deg < min_err:
                    min_err = err_deg
            return min_err
        elif sym_type == 'y-90':
            R_candidates = []
            R_candidates.append(R2)
            for angle in [90, 180, 270]:
                y_rot = Rotation.from_euler('zyx', [0, angle, 0], degrees=True).as_matrix()
                R_candidates.append(R2 @ y_rot)
            min_err = 180.0
            for R_cand in R_candidates:
                R_diff = R1 @ R_cand.T
                theta = np.arccos((np.trace(R_diff) - 1) / 2)
                err_deg = theta * 180 / np.pi
                if err_deg < min_err:
                    min_err = err_deg
            return min_err
        
        elif sym_type == 'sphere-symmetric':
            return 0.0
            
        else:
            raise NotImplementedError(f"Symmetry type {sym_type} not implemented.")

if __name__ == "__main__":
    import gorilla


    checkpoint_path = 'log/checkpoint.pth'
    data_dir = 'dataset/canoverse_200k/test'

    epoch = '45'
    logger = get_logger(epoch)
    logger.info("Starting PTS Test...")

    class Config:
        sample_num = 8192
        data_dir = data_dir
    config = Config()
    test_dataset = PTSTestDataset(config)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=lambda x: x)

    model = Net(resolution=64, ds_rate=2)
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint_path)
    model = model.cuda()
    model.eval()

    all_errs = []

    with torch.no_grad():

        classid_dir_map = {v: k for k, v in test_dataset.dir_classid_map.items()}
        total_samples = len(test_dataset)
        pbar = tqdm(total=total_samples, desc="PTS Test", ncols=100)
        for batch in test_loader:
            for ret_dict in batch:
                for key in ret_dict:
                    if isinstance(ret_dict[key], torch.Tensor):
                        ret_dict[key] = ret_dict[key].unsqueeze(0).cuda()
                pred_rotation = model(ret_dict)
                pred_rotation = pred_rotation['pred_rotation'].detach().cpu().numpy()
                gt_rotation = ret_dict['rotation_label'].detach().cpu().numpy()
                cat_id = int(ret_dict['category_label'].item())
                
                class_name = classid_dir_map.get(cat_id, "unknown")
                sym_type = test_dataset.cate_sym_map.get(class_name, None)
                if sym_type is not None:
                    err = cal_err_rot_sym_axis(pred_rotation[0], gt_rotation[0], sym_type)
                else:
                    err = ErrNdeg(pred_rotation[0], gt_rotation[0])
            
                all_errs.append(err)

                cate_name = classid_dir_map.get(cat_id, "unknown")

                npy_path = str(ret_dict['npy_path'])
                obj_id = Path(npy_path).parent.name

                logger.info(f"[{cate_name}][{obj_id}] Rotation Error: {err:.2f} degrees")
                
                pbar.update(1)
                pbar.set_postfix(avg_err=f"{np.mean(all_errs):.2f}")
        pbar.close()

    logger.info(f"Mean rotation error: {np.mean(all_errs):.2f} deg, Std: {np.std(all_errs):.2f} deg")
    print(f"Mean rotation error: {np.mean(all_errs):.2f} deg, Std: {np.std(all_errs):.2f} deg")

    # Error range statistics
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 160, 180]
    bin_names = [
        "0-5 deg", "5-10 deg", "10-15 deg", "15-20 deg", "20-25 deg", "25-30 deg", "30-35 deg", "35-40 deg",
        "40-45 deg", "45-50 deg", "50-55 deg", "55-60 deg", "60-160 deg", "160-180 deg"
    ]
    counts = [0] * (len(bins) - 1)
    for err in all_errs:
        for i in range(len(bins)-1):
            if bins[i] <= err < bins[i+1]:
                counts[i] += 1
                break

    total = len(all_errs)
    print("Error range statistics:")
    for name, count in zip(bin_names, counts):
        percent = count / total * 100 if total > 0 else 0
        print(f"{name}: {count} ({percent:.2f}%)")
        logger.info(f"{name}: {count} ({percent:.2f}%)")
