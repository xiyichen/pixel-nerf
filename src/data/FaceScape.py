import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor
import cv2
import json
import random

class FaceScapeDataset(torch.utils.data.Dataset):
    """
    Dataset from FaceScape
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        # self.base_path = path + "_" + stage
        # self.dataset_name = os.path.basename(path)

        # print("Loading FaceScape dataset", self.base_path, "name:", self.dataset_name)
        # self.stage = stage
        # assert os.path.exists(self.base_path)

        # self.intrins = sorted(
        #     glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        # )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = (128, 128)
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        
        uids = []
        if stage == 'train':
            for i in range(1, 360):
                if i in [122, 212, 340, 344] or i > 325:
                    continue
                for j in range(1,21,1):
                    if not os.path.isdir(f'/cluster/scratch/xiychen/data/facescape_color_calibrated/{str(i).zfill(3)}/{str(j).zfill(2)}'):
                        continue
                    uids.append(f'{str(i).zfill(3)}/{str(j).zfill(2)}')
        else:
            for i in range(1, 360):
                if i in [122, 212, 340, 344] or i > 325:
                    for j in range(1, 21, 1):
                        if not os.path.isdir(f'/cluster/scratch/xiychen/data/facescape_color_calibrated/{str(i).zfill(3)}/{str(j).zfill(2)}'):
                            continue
                        uids.append(f'{str(i).zfill(3)}/{str(j).zfill(2)}')
        self.uids = uids

        self.z_near = 0.8
        self.z_far = 1.8
        self.lindisp = False

    def __len__(self):
        return len(self.uids)
    
    def read_transparent_png(self, filename):
        image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        alpha_channel = image_4channel[:,:,3]
        rgb_channels = image_4channel[:,:,:3]

        # White Background Image
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Alpha factor
        alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        return (final_image).astype(np.uint8)

    def __getitem__(self, index):
        dir_path = f'/cluster/scratch/xiychen/data/facescape_color_calibrated/{self.uids[index]}'
        with open(os.path.join(dir_path, 'cameras.json'), 'r') as f:
            camera_dict = json.load(f)
        
        valid_views = []
        for view in camera_dict.keys():
            if os.path.isfile(os.path.join(dir_path, f'view_{str(view).zfill(5)}', 'rgba_colorcalib_v2.png')):
                valid_views.append(view)
        view_candidates = []
        for valid_view in valid_views:
            if abs(camera_dict[valid_view]['angles']['azimuth']) <= 90:
                view_candidates.append(valid_view)
        
        if len(view_candidates) < 16:
            dir_path = os.path.join(self.data_dir, '085/03')
            with open(os.path.join(dir_path, 'cameras.json'), 'r') as f:
                camera_dict = json.load(f)
            
            valid_views = []
            for view in camera_dict.keys():
                if os.path.isfile(os.path.join(dir_path, f'view_{str(view).zfill(5)}', 'rgba_colorcalib_v2.png')):
                    valid_views.append(view)
            view_candidates = []
            for valid_view in valid_views:
                if abs(camera_dict[valid_view]['angles']['azimuth']) <= 90:
                    view_candidates.append(valid_view)
        view_candidates = random.sample(view_candidates, 16)
        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        all_focals = []
        all_cs = []
        for view in view_candidates:
            img = self.read_transparent_png(os.path.join(dir_path, f'view_{str(view).zfill(5)}', 'rgba_colorcalib_v2.png'))[:,:,::-1].copy()
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = np.eye(4)
            pose[:3,:4] = camera_dict[view]['extrinsics']
            
            intrinsics = camera_dict[view]['intrinsics']
            all_focals.append(torch.tensor([intrinsics[0][0], intrinsics[1][1]]).float())
            all_cs.append(torch.tensor([intrinsics[0][2], intrinsics[1][2]]).float())
            pose = torch.from_numpy(pose).float()
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)
        all_focals = torch.stack(all_focals)
        all_cs = torch.stack(all_cs)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            all_focals *= scale
            all_cs *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            all_focals *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        all_focals = torch.tensor(all_focals, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": all_focals,
            "c": all_cs,
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
