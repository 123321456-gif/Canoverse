
import torch
from segment_anything import sam_model_registry, SamPredictor

from src.seg2d.glip_inference import glip_inference, load_model
from src.utils.h5_utils import save_masks_hdf5
from src.utils.time import timer




# 2D seg

class seg2d:
    # @timer
    def __init__(self, device, min_image_size=800):
        self.device = device
        self.min_image_size = min_image_size

        # loading glip
        print('[loading glip...]')
        config ="external_models/GLIP/configs/glip_Swin_L.yaml"
        weight_path = "./data/external_models/glip_large_model.pth"        
        self.glip_demo = load_model(config, weight_path, min_image_size=self.min_image_size)
        # loading sam
        print('[loading sam...]')
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = "./data/external_models/sam_vit_h_4b8939.pth"
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=torch.device("cuda:0"))
        self.sam_predictor = SamPredictor(sam)
        self.num_views = 10

    ####2D seg
    # @timer
    def seg_2D(self, support_path, part_names):
        # 2D seg
        # print("[glip infrence...]")
        masks = glip_inference(self.glip_demo, support_path, part_names, self.sam_predictor, num_views=self.num_views)
        # print('[finish 2Dseg...]')
        # 保存
        save_masks_hdf5(masks, support_path + '/glip_pred/masks.h5')
        return support_path + '/glip_pred/masks.h5'

if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_root = 'results/objaverse/newObj/results/boot/ins_0001/rot_0'

    seg2d = seg2d(device)
    # part_names = ['tail']
    part_names = ["sole", "laces", "heel"]
    seg2d.seg_2D(data_root, part_names)