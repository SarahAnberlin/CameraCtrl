import cv2
import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from torch.utils.data.dataset import Dataset
from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    # B, V, HW, 3
    # print(f"rays_o shape: {rays_o.shape}, rays_d shape: {rays_d.shape}")
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    # print(f"rays_dxo shape: {rays_dxo.shape}")
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


# data_root = '/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/re10k/'
# train_dir = 'train'
# test_dir = 'test'
# train_root = os.path.join(data_root, train_dir)
# test_root = os.path.join(data_root, test_dir)
# re10k_train_meta_json = '/hpc2hdd/home/hongfeizhang/dataset/realestate10k/re10k_train_meta.json'
# re10k_test_meta_json = '/hpc2hdd/home/hongfeizhang/dataset/realestate10k/re10k_test_meta.json'


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class RealEstate10KPose(Dataset):
    def __init__(
        self,
        split="train",
        sample_stride=8,
        minimum_sample_stride=1,
        sample_n_frames=21,
        relative_pose=True,
        sample_size=[320, 480],
        rescale_fxy=True,
        use_flip=False,
        no_extra_frame=True,
        return_caption=False,
    ):
        self.split = split
        self.return_caption = return_caption
        if self.split == "train":
            self.meta_json_path = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/train_meta.json"
            )
            self.data_root = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/train_scenes"
            )
            self.caption_json_path = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/train_captions.json"
            )
        else:
            self.meta_json_path = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/test_meta.json"
            )
            self.data_root = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/test_scenes"
            )
            self.caption_json_path = (
                "/hpc2hdd/home/hongfeizhang/dataset/realestate10k/test_captions.json"
            )
        with open(self.caption_json_path, "r") as f:
            self.captions = json.load(f)
        self.no_extra_frame = no_extra_frame
        with open(self.meta_json_path, "r") as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} samples from {self.meta_json_path}")

        self.relative_pose = relative_pose
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames

        self.length = len(self.dataset)

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.sample_size = sample_size

        if use_flip:
            pixel_transforms = [
                transforms.Resize(sample_size),
                RandomHorizontalFlipWithPose(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        else:
            pixel_transforms = [
                transforms.Resize(sample_size),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        # Always zero_init the first camera pose
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def decode_image(self, image_tensor):
        byte_data = image_tensor.numpy().tobytes()
        np_data = np.frombuffer(byte_data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img

    def get_batch(self, idx):
        #
        #   {
        #     "path": "01042371ee0b76ac.torch",
        #     "frame": 83,
        #     "height": 360,
        #     "width": 640,
        #     "key": "01042371ee0b76ac",
        #     "url": "https://www.youtube.com/watch?v=LiejcFr3v7c"
        #   },
        data_root = self.data_root
        current_sample_stride = self.sample_stride
        json_entry = self.dataset[idx]
        # print(
        #     f"Loading sample {idx}: {json_entry['key']} with {json_entry['frame']} frames")

        # Frame part
        total_frames = json_entry["frame"]
        key = json_entry["key"]
        video_caption = self.captions.get(key + ".mp4", [''])
        # if train random pick one, if test, first one 
        if self.split == "train":
            video_caption = random.choice(video_caption)
        else:
            if isinstance(video_caption, list):
                if len(video_caption) == 0:
                    video_caption = ['']
            elif isinstance(video_caption, str):
                video_caption = [video_caption]
            video_caption = video_caption[0]  # Use the first caption

        # print(f"Video caption for {key}: {video_caption}")
        if total_frames < self.sample_n_frames * self.minimum_sample_stride:
            raise ValueError(
                f"Total frames {total_frames} is less than sample_n_frames {self.sample_n_frames}"
            )

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(
                self.minimum_sample_stride, maximum_sample_stride
            )

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int
        )

        # Image and camera
        data = torch.load(os.path.join(data_root, json_entry["path"]))
        images = data["images"]
        cameras_info = data["cameras"]

        # Camera
        cameras_info = torch.concat(
            [torch.zeros(cameras_info.shape[0], 1), cameras_info], dim=1
        )
        cam_params = [Camera(cameras_info[indice]) for indice in frame_indices]

        pixel_values = [self.decode_image(image_tensor) for image_tensor in images]

        # image part
        pixel_values = np.stack(pixel_values, axis=0)  # [F, H, W, C]
        pixel_values = (
            torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        )  # [F, C, H, W]
        pixel_values = pixel_values / 255.0  # Normalize to [0, 1]
        # select the sampled frames
        pixel_values = pixel_values[frame_indices]

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:  # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:  # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

        intrinsics = np.asarray(
            [
                [
                    cam_param.fx * self.sample_size[1],
                    cam_param.fy * self.sample_size[0],
                    cam_param.cx * self.sample_size[1],
                    cam_param.cy * self.sample_size[0],
                ]
                for cam_param in cam_params
            ],
            dtype=np.float32,
        )

        intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array(
                [cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32
            )
        # [1, n_frame, 4, 4]
        c2w = torch.as_tensor(c2w_poses)[None]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(
                self.sample_n_frames, dtype=torch.bool, device=c2w.device
            )
        plucker_embedding = (
            ray_condition(
                intrinsics,
                c2w,
                self.sample_size[0],
                self.sample_size[1],
                device="cpu",
                flip_flag=flip_flag,
            )[0]
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return pixel_values, plucker_embedding, flip_flag,video_caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video, plucker_embedding, flip_flag = None, None, None
        while True:
            try:
                video, plucker_embedding, flip_flag,video_caption = self.get_batch(idx)
                break
            except Exception as e:
                # print(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, self.length - 1)
        # video,  plucker_embedding, flip_flag = self.get_batch(
        #     idx)

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            # video = self.pixel_transforms[2](video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        sample = {
            "images": video,  # F C H W format
            "control": torch.zeros_like(video),  # Placeholder for control
            # Align dimensions
            "camera_infos": plucker_embedding.permute(1, 0, 2, 3),
            "prompt": video_caption,
        }
        if self.no_extra_frame:
            sample["extra_images"] = None
            sample["extra_image_frame_index"] = None
        return sample

    def visualizevideo(self, idx, save_root="video", fixed_point_world=None):
        """Generate visualization video with camera projection overlay"""
        import cv2
        import numpy as np
        import os
        from tqdm import tqdm

        if fixed_point_world is None:
            fixed_point_world = np.array([0.0, 0.0, 0.0])
        assert fixed_point_world.shape == (3,), "fixed_point_world must be a 3D point."

        # Load sample data using existing get_batch logic
        json_entry = self.dataset[idx]
        data = torch.load(os.path.join(self.data_root, json_entry["path"]))
        images = data["images"]
        cameras_info = data["cameras"]
        clip_name = json_entry["key"]
        total_frames = json_entry["frame"]

        # Sample frame indices
        current_sample_stride = self.sample_stride
        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = max(1, maximum_sample_stride)

        print(
            f"Using sample stride: {current_sample_stride} for total frames: {total_frames}"
        )
        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = max(0, total_frames - cropped_length) // 2
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
        frame_indices = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int
        )

        # Decode images
        pixel_values = [self.decode_image(images[i]) for i in frame_indices]
        pixel_values = np.stack(pixel_values, axis=0)  # [F, H, W, C]
        pixel_values = (
            torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.0
        )

        # Camera parameters
        cameras_info = torch.concat(
            [torch.zeros(cameras_info.shape[0], 1), cameras_info], dim=1
        )
        cam_params = [Camera(cameras_info[i]) for i in frame_indices]

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:  # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:  # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]

        # Flipping
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool)

        # Apply pixel transforms
        if self.use_flip:
            images = self.pixel_transforms[0](pixel_values)
            images = self.pixel_transforms[1](images, flip_flag)
            # images = self.pixel_transforms[2](images)
        else:
            images = pixel_values
            for transform in self.pixel_transforms:
                images = transform(images)

        images_np = (
            ((images.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255)
            .clamp(0, 255)
            .numpy()
            .astype(np.uint8)
        )
        height, width = images_np.shape[1:3]

        # Create save directory
        os.makedirs(save_root, exist_ok=True)
        video_filename = f"realestate10k_pose_{idx}_{clip_name}.avi"
        video_path = os.path.join(save_root, video_filename)

        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height)
        )

        print(f"Generating visualization video for sample {idx} ({clip_name})")
        print(f"Video will be saved to: {video_path}")

        for frame_idx, img in enumerate(tqdm(images_np, desc="Writing frames")):
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cam_param = cam_params[frame_idx]

            K = np.array(
                [
                    [
                        cam_param.fx * self.sample_size[1],
                        0,
                        cam_param.cx * self.sample_size[1],
                    ],
                    [
                        0,
                        cam_param.fy * self.sample_size[0],
                        cam_param.cy * self.sample_size[0],
                    ],
                    [0, 0, 1],
                ]
            )
            w2c = cam_param.w2c_mat

            P_world_h = np.hstack([fixed_point_world, 1])
            P_cam_h = w2c @ P_world_h
            P_cam = P_cam_h[:3]

            if P_cam[2] > 0:
                p_img_h = K @ (P_cam / P_cam[2])
                u, v = int(p_img_h[0]), int(p_img_h[1])
                u = np.clip(u, 0, width - 1)
                v = np.clip(v, 0, height - 1)

                cv2.circle(img_bgr, (u, v), 5, (0, 0, 255), -1)
                cv2.line(img_bgr, (u - 10, v), (u + 10, v), (0, 255, 0), 1)
                cv2.line(img_bgr, (u, v - 10), (u, v + 10), (0, 255, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_bgr,
                f"Frame {frame_idx+1}/{self.sample_n_frames}",
                (10, 30),
                font,
                0.6,
                (255, 255, 255),
                1,
            )
            cam_pos = cam_param.c2w_mat[:3, 3]
            cv2.putText(
                img_bgr,
                f"Cam pos: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})",
                (10, height - 30),
                font,
                0.5,
                (255, 255, 255),
                1,
            )

            video_writer.write(img_bgr)

        video_writer.release()
        print(f"Visualization saved to: {video_path}")


if __name__ == "__main__":
    # Example usage
    save_root = "video/realestate10k_pose"
    os.makedirs(save_root, exist_ok=True)
    dataset = RealEstate10KPose(
        split="test",
        sample_stride=7,
        sample_n_frames=21,
        relative_pose=True,
        sample_size=[320, 480],
        rescale_fxy=False,
        use_flip=False,
    )

    def custom_collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            values = [d[key] for d in batch]

            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                collated[key] = values
            elif values[0] is None:
                collated[key] = None
            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(values[0])}")
        return collated

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    for data in dataset:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}, dtype={v.dtype}")
            elif isinstance(v, str):
                print(f"{k}: {v}, type={type(v)}")
            elif isinstance(v, list):
                print(f"{k}: {v}, length={len(v)}")
            else:
                print(f"{k}: {type(v)}")
            if k == "camera_infos":
                print(f"camera_infos min: {v.min()}, max: {v.max()}")
            elif k == "images":
                print(f"images min: {v.min()}, max: {v.max()}")
