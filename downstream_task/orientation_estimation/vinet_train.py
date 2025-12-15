import os
import sys
import argparse
import logging
import random

import torch
import gorilla
import json
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from utils.solver import Solver, get_logger
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from sym_process.sym_process_util import (
    calculate_connacial_rotation_ry,
    calculate_connacial_rotation_rz,
    calculate_connacial_rotation_rx,
    sphere_sym,
    y_180_sym,
    x_180_sym,
    z_180_sym,
    y_90_sym,
    sym_90_any_axis
)

import random

class PTSTrainingDataset(Dataset):
    def __init__(self,
            config, 
            resolution=64,
            ds_rate=2
    ):
        self.config = config

        self.resolution = resolution
        self.ds_rate = ds_rate
        self.sample_num = self.config.sample_num
        self.data_dir = config.data_dir
        
        self.cate_sym_map = json.load(open('sym_process/cate_sym_label.json'))
        pts_dir_path = config.data_dir
        if os.path.exists(f'{pts_dir_path}_cache'):
            self.npy_paths = json.load(open(f'{pts_dir_path}_cache/cached_npy_paths.json'))
            self.npy_paths = [os.path.join(pts_dir_path, p) for p in self.npy_paths]
        else:
            self.npy_paths = list(tqdm(Path(pts_dir_path).rglob('*.npy')))
            if os.path.exists(f'{pts_dir_path}_cache') is False:
                os.makedirs(f'{pts_dir_path}_cache')
            json.dump([str(Path(p).relative_to(pts_dir_path)) for p in self.npy_paths], open(f'{pts_dir_path}_cache/cached_npy_paths.json', 'w'))
            self.npy_paths = [str(p) for p in self.npy_paths]

        self.dir_classid_map = {cat: idx for idx, cat in enumerate(sorted(os.listdir(pts_dir_path)))}
        
        
        
        print('{} npys files are found.'.format(len(self.npy_paths)))
        if len(self.npy_paths) == 0:
            raise ValueError('No pts files found in {}'.format(pts_dir_path))
        
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        # if self.num_pts_per_epoch != -1:
        #     self.reset()
    def _process_symmetry(self, rotation: np.ndarray, cate: str):
        
        if cate not in self.cate_sym_map:
            return rotation, 1.0  
        sym_type = self.cate_sym_map[cate]
        if sym_type == 'y-180':
            rotation = y_180_sym(rotation)
            asym_flag = 0.0
        elif sym_type == 'x-180':
            rotation = x_180_sym(rotation)
            asym_flag = 0.0
        elif sym_type == 'z-180':
            rotation = z_180_sym(rotation)
            asym_flag = 0.0
        elif sym_type == '90-symmetric':
            rotation = sym_90_any_axis(rotation)
            asym_flag = 0.0
        elif sym_type == 'y-90':
            rotation = y_90_sym(rotation)
            asym_flag = 0.0
        elif sym_type == 'y-symmetric':
            rotation = calculate_connacial_rotation_ry(rotation)
            asym_flag = 0.0
        elif sym_type == 'z-symmetric':
            rotation = calculate_connacial_rotation_rz(rotation)
            asym_flag = 0.0
        elif sym_type == 'x-symmetric':
            rotation = calculate_connacial_rotation_rx(rotation)
            asym_flag = 0.0
        elif sym_type == 'sphere-symmetric':
            rotation = sphere_sym(rotation)
            asym_flag = 0.0
        else:
            raise NotImplementedError(f"Symmetry type {sym_type} not implemented.")
        return rotation, asym_flag
    
    def __len__(self):

        return len(self.npy_paths)*40
        
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

        rotation = R.random().as_matrix().astype(np.float32)
        
        aug_bb = np.random.uniform(0.8, 1.2, 3)
        points = points * aug_bb
        points = points / np.linalg.norm(points, axis=-1).max()

        points = points.astype(np.float32) @ rotation.T
        normals = normals.astype(np.float32) @ rotation.T


        translation = np.zeros((3,)).astype(np.float32)

        size = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).astype(np.float32)

        rotation, asym_flag = self._process_symmetry(rotation, dir_name)

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
        ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['size_label'] = torch.FloatTensor(size)

        ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
        ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
        ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
        ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)

        return ret_dict



def get_parser():
    parser = argparse.ArgumentParser(
        description="VI-Net")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/pts.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--mode",
                        type=str,
                        default="r",
                        help="[r|ts]")
    parser.add_argument("--checkpoint_epoch",
                        type=int,
                        default=-1,
                        help="checkpoint epoch: -1 / 0")
    parser.add_argument("--data_dir",type=str,default="dataset/canoverse_200k/train",help="data root dir")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dataset = args.dataset
    cfg.mode = args.mode
    cfg.gpus = args.gpus
    cfg.checkpoint_epoch = args.checkpoint_epoch
    cfg.data_dir = args.data_dir
    if cfg.mode == 'ts':
        cfg.log_dir = os.path.join('log', args.dataset, 'PN2')
    elif cfg.mode == 'r':
        cfg.log_dir = os.path.join('log', args.dataset, 'PTS_VI_Net')
    else:
        assert False, 'Wrong mode'

    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir("log/"+args.dataset):
        os.makedirs("log/"+args.dataset)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    from model.PTS_VI_Net import Net, Loss
    model = Net(cfg.resolution, cfg.ds_rate)
    
    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # loss
    loss = Loss(cfg.loss).cuda()    

    # dataloader
    dataset = PTSTrainingDataset(
        cfg.train_dataset,
        resolution = cfg.resolution,
        ds_rate = cfg.ds_rate,
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers= 8,
        shuffle=cfg.train_dataloader.shuffle,
        sampler=None,
        drop_last=cfg.train_dataloader.drop_last,
        pin_memory=cfg.train_dataloader.pin_memory,
        persistent_workers=True
    )

    dataloaders = {
        "train": dataloader,
    }

    # solver
    Trainer = Solver(model=model, loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     check_point_prefix = 'pts_vinet'
                     )
    Trainer.solve('pts_vinet')

    logger.info('\nFinish!\n')
