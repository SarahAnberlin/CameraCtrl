import os
import json
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
import random
import torch.nn.functional as F
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
# import dataset
from torch.utils.data import Dataset
from .graphics_utils import focal2fov
from .camera_utils import CameraInfo
import torchvision
import sys
import os
# import depth_pro
# depth_estimator, transform = depth_pro.create_model_and_transforms()


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "colmap/sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse/0")
        assert os.path.exists(colmap_dir), (
            f"COLMAP directory {colmap_dir} does not exist."
        )

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate(
                [np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array(
                    [cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array(
                    [cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert camtype == "perspective" or camtype == "fisheye", (
                f"Only perspective and fisheye cameras are supported, got {type_}"
            )

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (
                cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]
        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        print("factor", factor)
        print("self.extconf", self.extconf)
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        if os.path.exists(os.path.join(data_dir, "images")):
            colmap_image_dir = os.path.join(data_dir, "images")
        else:
            colmap_image_dir = os.path.join(data_dir, "images_4")
        # TODO
        image_dir = os.path.join(data_dir, "images_4" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f])
                       for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        # Dict[str, np.ndarray], image_name -> [M,]
        self.point_indices = point_indices
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert camera_id in self.params_dict, (
                f"Missing params for camera {camera_id}"
            )
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def filter_points(
        self, train_indices: List[int], bounding_boxes: List[Tuple[int, int, int, int]]
    ):
        """
        Filter points based on visibility in sampled views and bounding boxes.

        :param bounding_boxes: List of (x, y, w, h) for each sampled view
        """
        visible_points = set()
        for idx, bbox in zip(train_indices, bounding_boxes):
            worldtocam = np.linalg.inv(self.camtoworlds[idx])
            K = self.Ks_dict[self.camera_ids[idx]]
            x, y, w, h = bbox
            point_indices = self.point_indices[self.image_names[idx]]
            points_world = self.points[point_indices]
            points_cam = (worldtocam[:3, :3] @
                          points_world.T + worldtocam[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points_2d = points_proj[:, :2] / points_proj[:, 2:3]
            in_bbox = (
                (points_2d[:, 0] >= x)
                & (points_2d[:, 0] < x + w)
                & (points_2d[:, 1] >= y)
                & (points_2d[:, 1] < y + h)
                & (points_cam[:, 2] > 0)
            )
            visible_points.update(point_indices[in_bbox])

        # Update points and related attributes
        keep_indices = list(visible_points)
        self.points = self.points[keep_indices]
        self.points_err = self.points_err[keep_indices]
        self.points_rgb = self.points_rgb[keep_indices]

        # Update point_indices
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.point_indices = {
            k: np.array([old_to_new[idx] for idx in v if idx in old_to_new])
            for k, v in self.point_indices.items()
        }

    def filter_points_sparse(self, train_indices: List[int]):
        """
        Filter points based on visibility in sampled views and bounding boxes.

        :param bounding_boxes: List of (x, y, w, h) for each sampled view
        """
        visible_points = set()

        for idx in train_indices:
            point_indices = self.point_indices[self.image_names[idx]]
            visible_points.update(point_indices)

        keep_indices = list(visible_points)
        self.points = self.points[keep_indices]
        self.points_err = self.points_err[keep_indices]
        self.points_rgb = self.points_rgb[keep_indices]

        # Update point_indices
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.point_indices = {
            k: np.array([old_to_new[idx] for idx in v if idx in old_to_new])
            for k, v in self.point_indices.items()
        }


class ScenePose(Dataset):
    """A simple dataset class."""

    def __init__(
        self,
        args,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[List[int]] = [],
        # load_depths: bool = False,
        sparse_view: int = 0,
        extrapolator=None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        # self.load_depths = load_depths
        self.extrapolator = extrapolator
        self.args = args
        print(f"Datset args: {self.args}")
        indices = np.arange(len(self.parser.image_names))

        ########## mip360 for sparse view#########
        if self.args.outpaint_type == "sparse" and sparse_view > 0:
            split_file = os.path.join(
                self.parser.data_dir, f"train_test_split_{sparse_view}.json"
            )
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    print("split type: ", split)
                    split_data = json.load(f)
                    if split == "train":
                        self.indices = np.array(split_data["train_ids"])
                        self.parser.filter_points_sparse(self.indices)
                    else:
                        self.indices = np.array(split_data["test_ids"])
                    print(f"Loaded {split} indices from {split_file}")
        ######### Other dataset for dense view (crop)#########
        elif self.args.outpaint_type == "crop":
            if split == "train":
                factor = int(self.args.downsample_factor)
                self.indices = indices[::factor]  # Select every other index
                print("train indices", self.indices)
                self.parser.filter_points_sparse(self.indices)
            else:
                factor = int(self.args.downsample_factor)
                self.indices = indices[~np.isin(indices, indices[::factor])]
                print("test indices", self.indices)
        else:
            if split == 'train':
                print(
                    f"[Dataset] Using all {len(indices)} indices for training.")
                self.indices = indices
            elif split == 'test':
                print(
                    f"[Dataset] Using all {len(indices)} indices for testing.")
                self.indices = indices
            # # TODO
            # # completion
            # if split == "train":
            #     if sparse_view > 0:
            #         idx_sub = np.linspace(0, len(indices) - 1, sparse_view)
            #         idx_sub = [round(i) for i in idx_sub]
            #         self.indices = [self.indices[i] for i in idx_sub]
            #     else:
            #         factor = int(self.args.downsample_factor)
            #         # Select every other index
            #         self.indices = indices[::factor]
            #         print("train indices", self.indices)
            #         self.parser.filter_points_sparse(self.indices)
            #     self.parser.filter_points_sparse(self.indices)
            # else:
            #     self.indices = indices[indices % self.parser.test_every == 0]
            #     print("test indices", self.indices)

        if len(self.patch_size):
            if (
                self.args.outpaint_type == "crop"
            ):
                traj_path = "./trajectories.npy"
                assert os.path.exists(traj_path), (
                    f"Trajectory file {traj_path} does not exist."
                )
                self.trajectories = np.load(traj_path)
                bounding_boxes = [
                    (x, y, int(self.patch_size[0]), int(self.patch_size[1]))
                    for x, y in self.trajectories
                ]

            elif (
                "SMERF" in self.parser.data_dir
                or "tandt" in self.parser.data_dir
                or "TNT_GOF" in self.parser.data_dir
            ):
                print(f"??")
                traj_path = os.path.join(
                    self.parser.data_dir, "trajectories.npy")
                assert os.path.exists(traj_path), (
                    f"Trajectory file {traj_path} does not exist."
                )
                self.trajectories = np.load(traj_path)
                self.patch_size = [self.trajectories[0]
                                   [2], self.trajectories[0][3]]
                bounding_boxes = [
                    (x, y, int(self.patch_size[0]), int(self.patch_size[1]))
                    for x, y, _, _ in self.trajectories
                ]
            else:
                # generate a renadom trajectories
                self.generate_fixed_step_trajectory()
                # bounding_boxes = [
                #     (x, y, int(self.patch_size[0]), int(self.patch_size[1]))
                #     for x, y, _, _ in self.trajectories
                # ]

            # self.parser.filter_points(self.indices, bounding_boxes)

        self.load_image_into_memory()
        # if split == "train" and self.args.mono_depth:
        #     if "mip_360" in self.parser.data_dir and sparse_view > 0:
        #         self.generate_mono_depth(step=1)
        #     else:
        #         self.generate_mono_depth(step=self.args.num_frames)

    # def generate_mono_depth(self, step=1):
    #     print("Generating mono depth...")
    #     start_idx = 0
    #     total_frames = len(self.indices)
    #     for start_idx in range(0, total_frames, step):
    #         end_idx = min(start_idx + step, total_frames)
    #         # RGB batch
    #         # .half()  # [16, 3, H, W]
    #         rgb_batch = self.data_list[start_idx:end_idx]
    #         rgb_batch = [data["image"] for data in rgb_batch]
    #         rgb_batch = torch.stack(rgb_batch)
    #         rgb_batch = rgb_batch.permute(3, 0, 1, 2)
    #         print(rgb_batch.shape, rgb_batch.max(), rgb_batch.min())
    #         if rgb_batch.shape[2] < 540 and rgb_batch.shape[3] < 960:
    #             crop_height = self.args.diffusion_crop_height // 2
    #             crop_width = self.args.diffusion_crop_width // 2
    #             rgb_batch = rgb_batch[..., :crop_height, :crop_width]
    #         else:
    #             crop_height = self.args.diffusion_crop_height
    #             crop_width = self.args.diffusion_crop_width
    #             rgb_batch = rgb_batch[..., :crop_height, :crop_width]
    #         rgb_batch = (rgb_batch - 0.5) * 2
    #         rgb_batch = F.interpolate(
    #             rgb_batch.unsqueeze(0),
    #             size=(
    #                 self.args.num_frames,
    #                 self.args.diffusion_resize_height,
    #                 self.args.diffusion_resize_width,
    #             ),
    #             mode="trilinear",
    #             align_corners=False,
    #         ).squeeze(0)
    #         repaired_rgb, repaired_depth = self.process_batch(rgb_batch)
    #         repaired_depth = F.interpolate(
    #             repaired_depth.unsqueeze(0),
    #             size=(self.args.num_frames, crop_height, crop_width),
    #             mode="nearest",
    #         ).squeeze(0)

    #         for j in range(end_idx - start_idx):
    #             frame_idx = start_idx + j
    #             # Get depth for current frame
    #             depth_frame = repaired_depth[:, j]
    #             self.data_list[frame_idx]["mono_depth"] = (
    #                 depth_frame.detach().cpu().numpy()
    #             )  # Add depth to existing data dict

    #         # Clear GPU memory after processing each batch
    #         torch.cuda.empty_cache()

    #         # Print progress
    #         print(f"Processed frames {start_idx} to {end_idx - 1}")

    def process_batch(self, rgb_batch):
        depth_batch = torch.zeros(
            (1, rgb_batch.shape[1], rgb_batch.shape[2], rgb_batch.shape[3]),
            device=rgb_batch.device,
        )

        # Randomly select one frame from rgb_batch as reference
        rand_idx = random.randint(0, rgb_batch.shape[1] - 1)
        # Shape: [3, 1, H, W]
        ref_batch = rgb_batch[:, rand_idx: rand_idx + 1, :, :]

        repaired_rgb, repaired_depth, _, _ = self.extrapolator.repair(
            rgb_batch, depth_batch, ref_batch
        )
        repaired_rgb = repaired_rgb.cpu()
        repaired_depth = repaired_depth.cpu()

        torch.cuda.empty_cache()
        return repaired_rgb, repaired_depth

    def load_image_into_memory(self):
        print(f"Loading {len(self.indices)} images into memory...")
        print(
            f"Trajectories: {len(self.trajectories) if hasattr(self, 'trajectories') else 'Not set'}")
        self.data_list = []
        for i, index in enumerate(self.indices):
            image_path = self.parser.image_paths[index]
            image_name = self.parser.image_names[index]
            # Extract scene name and image name from path

            image = imageio.imread(self.parser.image_paths[index])[..., :3]
            height, width = image.shape[:2]
            camera_id = self.parser.camera_ids[index]
            K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
            params = self.parser.params_dict[camera_id]
            c2w = self.parser.camtoworlds[index]
            mask = self.parser.mask_dict[camera_id]
            w2c = np.linalg.inv(c2w)

            if len(params) > 0:
                # Images are distorted. Undistort them.
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y: y + h, x: x + w]

            # print(
            #     f"Self.trajectories: {self.trajectories if hasattr(self, 'trajectories') else 'Not set'}")
            if len(self.patch_size) > 0:
                # # Random crop.self.patch_size
                if len(self.trajectories[0]) == 4:
                    print(
                        f"Using trajectory with 4 elements: {self.trajectories[index]}")
                    x_start, y_start, _, _ = self.trajectories[index]
                else:
                    x_start, y_start = self.trajectories[index]
                image = image[
                    y_start: y_start + int(self.patch_size[1]),
                    x_start: x_start + int(self.patch_size[0]),
                ]
                # print(image.shape)
                K[0, 2] -= x_start
                K[1, 2] -= y_start

            h, w = image.shape[:2]

            cam_info = CameraInfo(
                uid=i,
                colmapid=index,
                K=K,
                w2c=w2c,
                image_name=image_name,
                image_path=image_path,
                width=w,
                height=h,
            )

            data = {
                "cam_info": cam_info,
                "image": torch.from_numpy(image) / 255.0,
                "image_name": image_name,
            }

            if mask is not None:
                data["mask"] = torch.from_numpy(mask)
            # print(f"If data has mask: {data.get('mask', None) is not None}")
            self.data_list.append(data)

    def visualize_camera_viewpoint(self, target_3d_point: np.ndarray):
        """
        Visualize each cropped image with an arrow pointing towards a fixed 3D world point.

        Args:
            target_3d_point: (3,) ndarray - the fixed 3D point in world coordinates
        """
        import os
        import cv2
        from tqdm import tqdm

        output_dir = "output_rgb_viewpoint"
        os.makedirs(output_dir, exist_ok=True)

        for idx, index in enumerate(tqdm(self.indices)):
            # Load image
            image = cv2.imread(self.parser.image_paths[index])
            if image is None:
                print(f"Failed to load image {self.parser.image_paths[index]}")
                continue

            # Crop image using trajectory
            if len(self.trajectories[0]) == 4:
                x_start, y_start, _, _ = self.trajectories[idx]
            else:
                x_start, y_start = self.trajectories[idx]
            image = image[
                y_start: y_start + int(self.patch_size[1]),
                x_start: x_start + int(self.patch_size[0]),
            ]
            h, w = image.shape[:2]

            # Get camera info
            K = self.parser.Ks_dict[self.parser.camera_ids[index]].copy()
            K[0, 2] -= x_start
            K[1, 2] -= y_start
            c2w = self.parser.camtoworlds[index]

            # Compute camera center in world coords
            cam_center = c2w[:3, 3]

            # Compute direction from camera center to target point
            direction = target_3d_point - cam_center
            direction = direction / np.linalg.norm(direction)

            # Transform direction to camera space
            R = c2w[:3, :3]
            direction_cam = R.T @ direction  # In camera's local frame

            if direction_cam[2] <= 0:
                print(f"Frame {idx}: target is behind the camera.")
                continue  # Point is behind the camera

            # Project direction to pixel
            pixel = K @ direction_cam
            pixel /= pixel[2]

            pixel_x, pixel_y = int(pixel[0]), int(pixel[1])
            center_x, center_y = w // 2, h // 2

            # Draw arrow
            arrowed_image = image.copy()
            cv2.arrowedLine(
                arrowed_image,
                (center_x, center_y),
                (pixel_x, pixel_y),
                (0, 0, 255),
                thickness=2,
                tipLength=0.1,
            )

            # Save image
            out_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(out_path, arrowed_image)

        print(f"Viewpoint visualization saved to {output_dir}")

    def generate_fixed_step_trajectory(self):
        original_state = np.random.get_state()
        np.random.seed()
        num_images = len(self.indices)
        image_0 = cv2.imread(self.parser.image_paths[0])
        h, w = image_0.shape[:2]
        max_height = (
            h - self.patch_size[1]
        )  # min(h - self.patch_size[1] for _, h in sizes)
        max_width = (
            w - self.patch_size[0]
        )  # min(w - self.patch_size[0] for w, _ in sizes)
        trajectories = []
        # for i in range(num_images):
        #     trajectories.append((0, 0))
        # self.trajectories = trajectories[:num_images]
        # return
        x = np.random.randint(0, max(w - int(self.patch_size[0]), 1))
        y = np.random.randint(0, max(h - int(self.patch_size[1]), 1))
        print("start: ", x, y)
        direction_x = 1
        direction_y = 1

        step_y = np.random.randint(5, 30)
        step_x = np.random.randint(5, 30)

        while len(trajectories) < num_images:
            new_x = x + direction_x * step_x
            new_y = y + direction_y * step_y
            if new_x >= max_width or new_y >= max_height or new_x < 0 or new_y < 0:
                if new_x >= max_width and new_y >= max_height:
                    direction_x = -1
                    direction_y = -1
                elif new_x < 0 and new_y >= max_height:
                    direction_x = 1
                    direction_y = -1
                elif new_x >= max_width and new_y < 0:
                    direction_x = -1
                    direction_y = 1
                elif new_x < 0 and new_y < 0:
                    direction_x = 1
                    direction_y = 1
                elif new_y >= max_height:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = -1
                elif new_x >= max_width:
                    direction_x = -1
                    direction_y = np.random.choice([-1, 1])
                elif new_y < 0:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = 1
                elif new_x < 0:
                    direction_x = 1
                    direction_y = np.random.choice([-1, 1])

                step_y = np.random.randint(5, 30)
                step_x = np.random.randint(5, 30)
                continue
            else:
                x, y = new_x, new_y
            trajectories.append((x, y))
        np.random.set_state(original_state)

        self.trajectories = trajectories[:num_images]
        self.visualize_trajectories_video()

    def visualize_trajectories(self):
        import os
        import cv2
        from tqdm import tqdm

        output_dir = "output_rgb"
        output_crop_dir = "output_rgb_crop"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_crop_dir, exist_ok=True)

        for idx, (image_path, (x, y)) in enumerate(
            tqdm(
                zip(self.parser.image_paths, self.trajectories),
                total=len(self.parser.image_paths),
            )
        ):
            # Read the image
            image = cv2.imread(image_path)

            # Draw the rectangle
            start_point = (x, y)
            end_point = (x + self.patch_size[0], y + self.patch_size[1])
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)

            # Save the image with rectangle
            output_path = os.path.join(
                output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image)

            # Crop and save the patch
            crop = image[y: y + self.patch_size[1], x: x + self.patch_size[0]]
            crop_output_path = os.path.join(
                output_crop_dir, os.path.basename(image_path)
            )
            cv2.imwrite(crop_output_path, crop)

        print(f"Visualized images saved in {output_dir}")

    def visualize_trajectories_video(self):
        import os
        import cv2
        from tqdm import tqdm

        output_dir = "output_rgb"
        output_crop_dir = "output_rgb_crop"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_crop_dir, exist_ok=True)

        # Get first image to determine size
        sample_image = cv2.imread(self.parser.image_paths[0])
        height, width = sample_image.shape[:2]
        crop_h, crop_w = self.patch_size[1], self.patch_size[0]

        # Define video writers
        video_path = os.path.join(output_dir, "visualized.avi")
        crop_video_path = os.path.join(output_crop_dir, "crops.avi")
        fps = 10  # Adjust FPS as needed
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (width, height))
        crop_writer = cv2.VideoWriter(
            crop_video_path, fourcc, fps, (crop_w, crop_h))

        for idx, (image_path, (x, y)) in enumerate(
            tqdm(zip(self.parser.image_paths, self.trajectories),
                 total=len(self.parser.image_paths))
        ):
            image = cv2.imread(image_path)

            # Draw the rectangle
            start_point = (x, y)
            end_point = (x + self.patch_size[0], y + self.patch_size[1])
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)

            # Write image to video
            video_writer.write(image)

            # Crop and write to crop video
            crop = image[y: y + self.patch_size[1], x: x + self.patch_size[0]]
            crop_writer.write(crop)

        video_writer.release()
        crop_writer.release()

        print(f"Saved video to {video_path}")
        print(f"Saved crop video to {crop_video_path}")

    def visualize_point_projection_video(self, fixed_point_world: np.ndarray, output_path="projection_video.avi"):
        """
        在 crop 后的每帧图像上画出从图像中心指向固定3D点在图像中的投影的箭头，生成视频。

        Args:
            fixed_point_world: np.array, shape (3,), 3D点世界坐标
            output_path: str, 视频保存路径
        """
        import cv2
        import numpy as np
        from tqdm import tqdm
        import os

        assert fixed_point_world.shape == (3,), "fixed_point_world应为3维坐标"

        h, w = self.patch_size[1], self.patch_size[0]

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 10
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for data in tqdm(self.data_list, desc="Visualizing projection video"):
            # 取图像，转成BGR uint8
            img = (data["image"].numpy() * 255).astype(np.uint8)
            if img.shape[0] == 3:  # C,H,W
                img = np.transpose(img, (1, 2, 0))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cam_info = data["cam_info"]
            K = cam_info.K
            w2c = cam_info.w2c

            # 把固定点投影到像素坐标
            P_world_h = np.hstack([fixed_point_world, 1])  # 齐次坐标 (4,)
            P_cam_h = w2c @ P_world_h  # 相机坐标齐次
            P_cam = P_cam_h[:3]

            if P_cam[2] <= 0:
                # 点在相机后面，不绘制箭头
                video_writer.write(img_bgr)
                continue

            # 投影到像素平面
            p_img_h = K @ (P_cam / P_cam[2])
            u, v = int(p_img_h[0]), int(p_img_h[1])

            # 图像中心
            center_x, center_y = w // 2, h // 2

            # 边界检查，防止越界
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)

            # 画箭头：绿色，起点图像中心，终点为投影点
            cv2.arrowedLine(
                img_bgr,
                (center_x, center_y),
                (u, v),
                color=(0, 255, 0),
                thickness=2,
                tipLength=0.2,
            )

            # 画投影点一个红色圆圈
            cv2.circle(img_bgr, (u, v), radius=5,
                       color=(0, 0, 255), thickness=-1)

            video_writer.write(img_bgr)

        video_writer.release()
        print(f"Projection video saved to {output_path}")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item: int) -> Dict[str, Any]:
            return self.data_list[item]


class ScenesDataset(Dataset):
    """A simple dataset class."""

    def __init__(
        self,
        args,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[List[int]] = [],
        # load_depths: bool = False,
        sparse_view: int = 0,
        extrapolator=None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        # self.load_depths = load_depths
        self.extrapolator = extrapolator
        self.args = args
        print(f"Datset args: {self.args}")
        indices = np.arange(len(self.parser.image_names))

        if split == 'train':
            print(
                f"[Dataset] Using all {len(indices)} indices for training.")
            self.indices = indices
        elif split == 'test':
            print(
                f"[Dataset] Using all {len(indices)} indices for testing.")
            self.indices = indices

        self.generate_fixed_step_trajectory()

        self.load_image_into_memory()

    def __getitem__(self, item: int):
        pass

    def __len__(self):
        return len(self.indices)

    def process_batch(self, rgb_batch):
        depth_batch = torch.zeros(
            (1, rgb_batch.shape[1], rgb_batch.shape[2], rgb_batch.shape[3]),
            device=rgb_batch.device,
        )

        # Randomly select one frame from rgb_batch as reference
        rand_idx = random.randint(0, rgb_batch.shape[1] - 1)
        # Shape: [3, 1, H, W]
        ref_batch = rgb_batch[:, rand_idx: rand_idx + 1, :, :]

        repaired_rgb, repaired_depth, _, _ = self.extrapolator.repair(
            rgb_batch, depth_batch, ref_batch
        )
        repaired_rgb = repaired_rgb.cpu()
        repaired_depth = repaired_depth.cpu()

        torch.cuda.empty_cache()
        return repaired_rgb, repaired_depth

    def load_image_into_memory(self):
        print(f"Loading {len(self.indices)} images into memory...")
        print(
            f"Trajectories: {len(self.trajectories) if hasattr(self, 'trajectories') else 'Not set'}")
        self.data_list = []
        for i, index in enumerate(self.indices):
            image_path = self.parser.image_paths[index]
            image_name = self.parser.image_names[index]
            # Extract scene name and image name from path

            image = imageio.imread(self.parser.image_paths[index])[..., :3]
            height, width = image.shape[:2]
            camera_id = self.parser.camera_ids[index]
            K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
            params = self.parser.params_dict[camera_id]
            c2w = self.parser.camtoworlds[index]
            mask = self.parser.mask_dict[camera_id]
            w2c = np.linalg.inv(c2w)

            if len(params) > 0:
                # Images are distorted. Undistort them.
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y: y + h, x: x + w]

            # print(
            #     f"Self.trajectories: {self.trajectories if hasattr(self, 'trajectories') else 'Not set'}")
            if len(self.patch_size) > 0:
                # # Random crop.self.patch_size
                if len(self.trajectories[0]) == 4:
                    print(
                        f"Using trajectory with 4 elements: {self.trajectories[index]}")
                    x_start, y_start, _, _ = self.trajectories[index]
                else:
                    x_start, y_start = self.trajectories[index]
                image = image[
                    y_start: y_start + int(self.patch_size[1]),
                    x_start: x_start + int(self.patch_size[0]),
                ]
                # print(image.shape)
                K[0, 2] -= x_start
                K[1, 2] -= y_start

            h, w = image.shape[:2]

            cam_info = CameraInfo(
                uid=i,
                colmapid=index,
                K=K,
                w2c=w2c,
                image_name=image_name,
                image_path=image_path,
                width=w,
                height=h,
            )

            data = {
                "cam_info": cam_info,
                "image": torch.from_numpy(image) / 255.0,
                "image_name": image_name,
            }

            if mask is not None:
                data["mask"] = torch.from_numpy(mask)
            # print(f"If data has mask: {data.get('mask', None) is not None}")
            self.data_list.append(data)

        """
        Visualize each cropped image with an arrow pointing towards a fixed 3D world point.

        Args:
            target_3d_point: (3,) ndarray - the fixed 3D point in world coordinates
        """
        import os
        import cv2
        from tqdm import tqdm

        output_dir = "output_rgb_viewpoint"
        os.makedirs(output_dir, exist_ok=True)

        for idx, index in enumerate(tqdm(self.indices)):
            # Load image
            image = cv2.imread(self.parser.image_paths[index])
            if image is None:
                print(f"Failed to load image {self.parser.image_paths[index]}")
                continue

            # Crop image using trajectory
            if len(self.trajectories[0]) == 4:
                x_start, y_start, _, _ = self.trajectories[idx]
            else:
                x_start, y_start = self.trajectories[idx]
            image = image[
                y_start: y_start + int(self.patch_size[1]),
                x_start: x_start + int(self.patch_size[0]),
            ]
            h, w = image.shape[:2]

            # Get camera info
            K = self.parser.Ks_dict[self.parser.camera_ids[index]].copy()
            K[0, 2] -= x_start
            K[1, 2] -= y_start
            c2w = self.parser.camtoworlds[index]

            # Compute camera center in world coords
            cam_center = c2w[:3, 3]

            # Compute direction from camera center to target point
            direction = target_3d_point - cam_center
            direction = direction / np.linalg.norm(direction)

            # Transform direction to camera space
            R = c2w[:3, :3]
            direction_cam = R.T @ direction  # In camera's local frame

            if direction_cam[2] <= 0:
                print(f"Frame {idx}: target is behind the camera.")
                continue  # Point is behind the camera

            # Project direction to pixel
            pixel = K @ direction_cam
            pixel /= pixel[2]

            pixel_x, pixel_y = int(pixel[0]), int(pixel[1])
            center_x, center_y = w // 2, h // 2

            # Draw arrow
            arrowed_image = image.copy()
            cv2.arrowedLine(
                arrowed_image,
                (center_x, center_y),
                (pixel_x, pixel_y),
                (0, 0, 255),
                thickness=2,
                tipLength=0.1,
            )

            # Save image
            out_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(out_path, arrowed_image)

        print(f"Viewpoint visualization saved to {output_dir}")

    def generate_fixed_step_trajectory(self):
        original_state = np.random.get_state()
        np.random.seed()
        num_images = len(self.indices)
        image_0 = cv2.imread(self.parser.image_paths[0])
        h, w = image_0.shape[:2]
        max_height = (
            h - self.patch_size[1]
        )  # min(h - self.patch_size[1] for _, h in sizes)
        max_width = (
            w - self.patch_size[0]
        )  # min(w - self.patch_size[0] for w, _ in sizes)
        trajectories = []
        # for i in range(num_images):
        #     trajectories.append((0, 0))
        # self.trajectories = trajectories[:num_images]
        # return
        x = np.random.randint(0, max(w - int(self.patch_size[0]), 1))
        y = np.random.randint(0, max(h - int(self.patch_size[1]), 1))
        print("start: ", x, y)
        direction_x = 1
        direction_y = 1

        step_y = np.random.randint(5, 30)
        step_x = np.random.randint(5, 30)

        while len(trajectories) < num_images:
            new_x = x + direction_x * step_x
            new_y = y + direction_y * step_y
            if new_x >= max_width or new_y >= max_height or new_x < 0 or new_y < 0:
                if new_x >= max_width and new_y >= max_height:
                    direction_x = -1
                    direction_y = -1
                elif new_x < 0 and new_y >= max_height:
                    direction_x = 1
                    direction_y = -1
                elif new_x >= max_width and new_y < 0:
                    direction_x = -1
                    direction_y = 1
                elif new_x < 0 and new_y < 0:
                    direction_x = 1
                    direction_y = 1
                elif new_y >= max_height:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = -1
                elif new_x >= max_width:
                    direction_x = -1
                    direction_y = np.random.choice([-1, 1])
                elif new_y < 0:
                    direction_x = np.random.choice([-1, 1])
                    direction_y = 1
                elif new_x < 0:
                    direction_x = 1
                    direction_y = np.random.choice([-1, 1])

                step_y = np.random.randint(5, 30)
                step_x = np.random.randint(5, 30)
                continue
            else:
                x, y = new_x, new_y
            trajectories.append((x, y))
        np.random.set_state(original_state)

        self.trajectories = trajectories[:num_images]


if __name__ == "__main__":
    # Example usage
    meta_json = '/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/camera_data_paths.json'
    meta_paths = []
    with open(meta_json, 'r') as f:
        # readlines
        meta_paths = f.readlines()
    print(f"Total scenes : {len(meta_paths)}")

    class Args:
        def __init__(self):
            self.outpaint_type = "none"
            self.mono_depth = False

    args = Args()
    for meta_path in meta_paths:
        data_dir = os.path.dirname(meta_path.strip())
        parser = Parser(data_dir, factor=1, normalize=True)
        dataset = ScenesDataset(args=args, parser=parser,
                                split="train",
                                patch_size=[720, 480])
        # 这里调用刚写的函数，指定你想观察的3D点坐标
        fixed_point = np.array([0.0, 0.0, 0.0])  # 比如世界坐标原点
        base_name = os.path.basename(data_dir)
        save_root = 'output_projection_videos'
        os.makedirs(save_root, exist_ok=True)
        output_video_path = os.path.join(
            save_root, f"{base_name}_projection_video.avi")
