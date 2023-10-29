"""Data parser for Neural 3D Video Synthesis (https://arxiv.org/abs/2103.02597) Dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Union

import cv2
import h5py
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (DataParser,
                                                         DataParserConfig,
                                                         DataparserOutputs)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DyNeRFDataParserConfig(DataParserConfig):
    """Neural 3D Video Synthesis Dataset parser config"""

    _target: Type = field(default_factory=lambda: DyNeRF)
    data: Path = Path("data/dynerf/flame_salmon_1")
    """Directory specifying location of data."""
    downscale_factor: int = 2
    """Downsampling factor for videoes. Original resolution is 2704x2028."""
    load_every: int = 1
    """Frame interval."""


@dataclass
class DyNeRF(DataParser):
    """Neural 3D Video Synthesis Dataset"""

    config: DyNeRFDataParserConfig
    includes_time: bool = True

    def __init__(self, config: DyNeRFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.downscale_factor: int = config.downscale_factor
        self.load_every: int = config.load_every

    def process_frames(self, camera_id: int) -> List[Path]:
        """Read video and extract frames."""
        video_path = self.data / f"cam{camera_id:02d}.mp4"
        out_folder = self.data / f"x{self.downscale_factor}" / f"cam{camera_id:02d}"
        if not out_folder.exists():
            CONSOLE.print(f"Downscale {video_path} to {out_folder}")
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), f"Unable to open {video_path}"

            out_folder.mkdir(parents=True)
            frame_id = 0
            ret, frame = cap.read()
            ori_h, ori_w = frame.shape[:2]
            h, w = ori_h // self.downscale_factor, ori_w // self.downscale_factor
            while ret:
                out_path = out_folder / f"{frame_id:05d}.png"
                cv2.imwrite(str(out_path), cv2.resize(frame, (w, h)))
                frame_id += 1
                ret, frame = cap.read()
        all_frames = len(list(out_folder.iterdir()))
        return [out_folder / f"{i:05d}.png" for i in range(0, all_frames, self.load_every)]

    def _generate_dataparser_outputs(
            self,
            split: str = "train",
    ) -> DataparserOutputs:
        # load poses
        poses_bounds = np.load(self.data / "poses_bounds.npy")
        poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)  # (num_cameras, 3, 5)
        heights = poses_hwf[:, 0, -1] / self.downscale_factor
        widths = poses_hwf[:, 1, -1] / self.downscale_factor
        focal = poses_hwf[:, 2, -1] / self.downscale_factor  # (num_cameras,)
        poses = poses_hwf[:, :3, :4]  # (num_cameras, 3, 4)
        # switch axis (https://docs.nerf.studio/quickstart/data_conventions.html)
        # down-right-back (LLFF) => right-up-back
        poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:4]], axis=-1)

        if split == "train":
            camera_ids = np.arange(1, poses.shape[0])
            if "coffee_martini" in str(self.data):
                # camera-12 in coffee_martini is unsynchronized
                camera_ids = np.setdiff1d(camera_ids, 12)
        elif split in ["val", "test"]:
            camera_ids = np.array([0])  # use camera-0 for testing
        else:
            raise NotImplementedError

        # load images
        _image_paths = [self.process_frames(i) for i in camera_ids]  # (num_cameras, num_frames)
        aligned_image_paths: List[List[Path]] = list(zip(*_image_paths))  # (num_frames, num_cameras)
        num_frames = len(aligned_image_paths)
        num_cameras = len(aligned_image_paths[0])
        CONSOLE.print(f"Loaded {num_frames} frames x {num_cameras} cameras")

        cameras = Cameras(  # (num_frames, num_cameras, ...)
            camera_to_worlds=torch.tensor(poses[None, camera_ids, :, :]).expand(num_frames, num_cameras, 3, 4),
            fx=torch.tensor(focal[None, camera_ids, None]).expand(num_frames, num_cameras, 1),
            fy=torch.tensor(focal[None, camera_ids, None]).expand(num_frames, num_cameras, 1),
            cx=torch.tensor(widths[None, camera_ids, None]).expand(num_frames, num_cameras, 1) / 2,
            cy=torch.tensor(heights[None, camera_ids, None]).expand(num_frames, num_cameras, 1) / 2,
            width=torch.tensor(widths[None, camera_ids, None]).expand(num_frames, num_cameras, 1),
            height=torch.tensor(heights[None, camera_ids, None]).expand(num_frames, num_cameras, 1),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.linspace(0, 1, num_frames)[:, None, None].expand(num_frames, num_cameras, 1),
        )
        assert cameras.shape == (num_frames, num_cameras)
        if split in ["val", "test"]:
            cameras = cameras.flatten()

        return DataparserOutputs(
            image_filenames=sum(aligned_image_paths, []),  # Flatten to List[Path]
            cameras=cameras,
            metadata=dict(
                camera_ids=camera_ids,
                heights=heights[camera_ids],
                widths=widths[camera_ids]
            )
        )
    
    def precompute_isg(
            self,
            isg_cache: Path,
            dataparser_outputs: DataparserOutputs,
            isg_gamma: float = 2e-2,
            precompute_device: Union[torch.device, str] = "cpu",
    ):
        num_frames, num_cameras = dataparser_outputs.cameras.shape
        heights = dataparser_outputs.cameras.height[0, :, 0]  # select first-frame for each cameras
        widths = dataparser_outputs.cameras.width[0, :, 0]  # select first-frame for each cameras
        aligned_image_paths = np.array(dataparser_outputs.image_filenames).reshape(num_frames, num_cameras)

        compute_isg = False
        if isg_cache.exists():
            CONSOLE.print(f"ISG cache found: {isg_cache}")
            with h5py.File(isg_cache) as f:
                if f["gamma"] != isg_gamma:
                    CONSOLE.print("Mismatched ISG gamma. Needs re-compute ISG")
                    compute_isg = True
        else:
            compute_isg = True
        if compute_isg:
            CONSOLE.print("Computing ISG. It may take much memory...")
            with h5py.File(isg_cache, "w") as f:
                f.create_dataset("gamma", data=isg_gamma)
                g = f.create_group("weights")
                for i in range(num_cameras):
                    height = int(heights[i])
                    width = int(widths[i])
                    CONSOLE.print(f"Computing ISG of {aligned_image_paths[0, i].parent}")
                    imgs = torch.empty((num_frames, height, width, 3), dtype=torch.float32)
                    for j in range(num_frames):
                        imgs[j] = cv2.imread(str(aligned_image_paths[j,i])).astype(np.float32) / 255.
                    imgs = imgs.to(precompute_device)

                    diffsq = (imgs - torch.median(imgs, dim=0)).square()
                    psi = diffsq / (diffsq + isg_gamma ** 2)
                    isg_weights = psi.abs().sum(-1) / 3  # (num_frames, height, width)
                    g.create_dataset(str(i), data=isg_weights.cpu().numpy(), chunks=(1, height, width))

    def precompute_ist(
            self,
            ist_cache: Path,
            dataparser_outputs: DataparserOutputs,
            ist_alpha: float = 0.1,
            ist_shift: int = 25,
            precompute_device: Union[torch.device, str] = "cpu"
    ):
        num_frames, num_cameras = dataparser_outputs.cameras.shape
        heights = dataparser_outputs.metadata["heights"]
        widths = dataparser_outputs.metadata["widths"]
        aligned_image_paths = np.array(dataparser_outputs.image_filenames).reshape(num_frames, num_cameras)

        compute_ist = False
        if ist_cache.exists():
            CONSOLE.print(f"IST cache found: {ist_cache}")
            with h5py.File(ist_cache) as f:
                if f["alpha"] != ist_alpha or f["shift"] != ist_shift:
                    CONSOLE.print("Mismatched IST hyperparams. Needs re-compute IST")
                    compute_ist = True
        else:
            compute_ist = True
        if compute_ist:
            CONSOLE.print("Computing IST. It may take much memory...")
            with h5py.File(ist_cache, "w") as f:
                f.create_dataset("alpha", data=ist_alpha)
                f.create_dataset("shift", data=ist_shift)
                g = f.create_group("weights")
                for i in range(num_cameras):
                    height = int(heights[i])
                    width = int(widths[i])
                    CONSOLE.print(f"Computing IST of {aligned_image_paths[0,i].parent}")
                    imgs = torch.empty((num_frames, height, width, 3), dtype=torch.float32)
                    for j in range(num_frames):
                        imgs[j] = cv2.imread(str(aligned_image_paths[j,i])).astype(np.float32) / 255.
                    imgs = imgs.to(precompute_device)

                    max_diff = torch.zeros_like(imgs[0])
                    for shift in range(1, ist_shift+1):
                        shift_left = torch.cat([imgs[shift:], torch.zeros_like(imgs[:shift])], dim=0)
                        shift_right = torch.cat([torch.zeros_like(imgs[:shift]), imgs[:-shift]], dim=0)
                        mymax = torch.maximum(torch.abs_(shift_left - imgs), torch.abs_(shift_right - imgs))
                        max_diff = torch.maximum(max_diff, mymax)
                    max_diff = torch.sum(max_diff, dim=-1) / 3  # (num_frames, height, width)
                    max_diff = max_diff.clamp_(min=ist_alpha)
                    g.create_dataset(str(i), data=max_diff.cpu().numpy(), chunks=(1, height, width))
