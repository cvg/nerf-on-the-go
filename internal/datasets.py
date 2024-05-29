# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different datasets implementation plus a general port for all the datasets."""

import abc
import copy
import json
import os
from os import path
import queue
import threading
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

import cv2
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
import jax
import numpy as np
from PIL import Image

# This is ugly, but it works.
import sys
sys.path.insert(0,'internal/pycolmap')
sys.path.insert(0,'internal/pycolmap/pycolmap')
import pycolmap
import math


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'on-the-go': Onthego,
  }
  return dataset_dict[config.dataset_loader](split, train_dir, config)


class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

  def process(
      self
  ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, Optional[Mapping[
      Text, float]], camera_utils.ProjectionType]:
    """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

    self.load_cameras()
    self.load_images()
    # self.load_points3D()  # For now, we do not need the point cloud data.

    # Assume shared intrinsics between all cameras.
    cam = self.cameras[1]

    # Extract focal lengths and principal point parameters.
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

    # Extract extrinsic matrices in world-to-camera format.
    imdata = self.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
      params = None
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 1 or type_ == 'PINHOLE':
      params = None
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    if type_ == 2 or type_ == 'SIMPLE_RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 3 or type_ == 'RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 4 or type_ == 'OPENCV':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['p1'] = cam.p1
      params['p2'] = cam.p2
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['k3'] = cam.k3
      params['k4'] = cam.k4
      camtype = camera_utils.ProjectionType.FISHEYE

    return names, poses, pixtocam, params, camtype


def load_blender_posedata(data_dir, split=None):
  """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
  suffix = '' if split is None else f'_{split}'
  pose_file = path.join(data_dir, f'transforms{suffix}.json')
  with utils.open_file(pose_file, 'r') as fp:
    meta = json.load(fp)
  names = []
  poses = []
  for _, frame in enumerate(meta['frames']):
    filepath = os.path.join(data_dir, frame['file_path'])
    if utils.file_exists(filepath):
      names.append(frame['file_path'].split('/')[-1])
      poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
    else:
      print('read unexpected file path:', filepath)
  poses = np.stack(poses, axis=0)
  w = meta['w']
  h = meta['h']
  cx = meta['cx'] if 'cx' in meta else w / 2.
  cy = meta['cy'] if 'cy' in meta else h / 2.
  if 'fl_x' in meta:
    fx = meta['fl_x']
  else:
    fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
  if 'fl_y' in meta:
    fy = meta['fl_y']
  else:
    fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))
  pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
  coeffs = ['k1', 'k2', 'p1', 'p2']
  if not any([c in meta for c in coeffs]):
    params = None
  else:
    params = {c: (meta[c] if c in meta else 0.) for c in coeffs}
  camtype = camera_utils.ProjectionType.PERSPECTIVE
  return names, poses, pixtocam, params, camtype


class Dataset(threading.Thread, metaclass=abc.ABCMeta):
  """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    metadata: dict, optional metadata for raw datasets.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

  def __init__(self,
               split: str,
               data_dir: str,
               config: configs.Config):
    super().__init__()

    # Initialize attributes
    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self._patch_size = np.maximum(config.patch_size, 1)
    self._batch_size = config.batch_size // jax.process_count()
    if self._patch_size**2 > self._batch_size:
      raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                       f'per-process batch size {self._batch_size}')
    self._batching = utils.BatchingMethod(config.batching)
    self._load_features = config.compute_feature_metrics
    self._test_camera_idx = 0
    self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
    self._apply_bayer_mask = config.apply_bayer_mask
    self._cast_rays_in_train_step = config.cast_rays_in_train_step
    self._render_spherical = False

    self.split = utils.DataSplit(split)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.render_path = config.render_path

    self.distortion_params = None
    self.disp_images = None
    self.normal_images = None
    self.features = None
    self.assignments = None
    self.alphas = None
    self.poses = None
    self.pixtocam_ndc = None
    self.metadata = None
    self.camtype = camera_utils.ProjectionType.PERSPECTIVE
    self.exposures = None
    self.render_exposures = None

    # Providing type comments for these attributes, they must be correctly
    # initialized by _load_renderings() (see docstring) in any subclass.
    self.images: np.ndarray = None
    self.camtoworlds: np.ndarray = None
    self.pixtocams: np.ndarray = None
    self.height: int = None
    self.width: int = None
    self.dilate = config.dilate
    self.is_render = config.is_render
    self.feat_rate = config.feat_rate

    # Load data from disk using provided config parameters.
    self._load_renderings(config)

    if self.render_path:
      if config.render_path_file is not None:
        with utils.open_file(config.render_path_file, 'rb') as fp:
          render_poses = np.load(fp)
        self.camtoworlds = render_poses
      if config.render_resolution is not None:
        self.width, self.height = config.render_resolution
      if config.render_focal is not None:
        self.focal = config.render_focal
      if config.render_camtype is not None:
        if config.render_camtype == 'pano':
          self._render_spherical = True
        else:
          self.camtype = camera_utils.ProjectionType(config.render_camtype)

      self.distortion_params = None
      self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                 self.height)

    self._n_examples = self.camtoworlds.shape[0]

    self.cameras = (self.pixtocams,
                    self.camtoworlds,
                    self.distortion_params,
                    self.pixtocam_ndc)


    # Seed the queue with one batch to avoid race condition.
    if self.split == utils.DataSplit.TRAIN:
      self._next_fn = self._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self._queue.get()
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      # Do NOT move test `rays` to device, since it may be very large.
      return x

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      return jax.device_put(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())

  @property
  def size(self):
    return self._n_examples

  @abc.abstractmethod
  def _load_renderings(self, config):
    """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      disp_images: [N, height, width] array for depth data (optional).
      normal_images: [N, height, width, 3] array for normals (optional).
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

  def _make_ray_batch(self,
                      pix_x_int: np.ndarray,
                      pix_y_int: np.ndarray,
                      cam_idx: Union[np.ndarray, np.int32],
                      lossmult: Optional[np.ndarray] = None
                      ) -> utils.Batch:
    """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
        'near': broadcast_scalar(self.near),
        'far': broadcast_scalar(self.far),
        'cam_idx': broadcast_scalar(cam_idx),
    }
    # Collect per-camera information needed for each ray.
    if self.metadata is not None:
      # Exposure index and relative shutter speed, needed for RawNeRF.
      for key in ['exposure_idx', 'exposure_values']:
        idx = 0 if self.render_path else cam_idx
        ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
    if self.exposures is not None:
      idx = 0 if self.render_path else cam_idx
      ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
    if self.render_path and self.render_exposures is not None:
      ray_kwargs['exposure_values'] = broadcast_scalar(
          self.render_exposures[cam_idx])
    if self._load_features:
      if self.is_render:
        features = np.zeros((pix_x_int.shape[0], pix_x_int.shape[1], self.features.shape[-1]))
      else:
        assignments = self.assignments[cam_idx, pix_y_int, pix_x_int]
        dim0, _, _, dim3 = self.features.shape
        features = self.features.reshape(dim0, -1, dim3)[cam_idx, assignments]

    pixels = utils.Pixels(pix_x_int, pix_y_int, features=features, **ray_kwargs)
    if self._cast_rays_in_train_step and self.split == utils.DataSplit.TRAIN:
      # Fast path, defer ray computation to the training loop (on device).
      rays = pixels
    else:
      # Slow path, do ray computation using numpy (on CPU).
      rays = camera_utils.cast_ray_batch(
          self.cameras, pixels, self.camtype, xnp=np)

    # Create data batch.
    batch = {}
    batch['rays'] = rays
    if not self.render_path:
      batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
    return utils.Batch(**batch)

  def _next_train(self) -> utils.Batch:
    """Sample next training batch (random rays)."""
    # We assume all images in the dataset are the same resolution, so we can use
    # the same width/height for sampling all pixels coordinates in the batch.
    # Batch/patch sampling parameters.
    num_patches = self._batch_size // self._patch_size ** 2 * self.dilate**2
    lower_border = self._num_border_pixels_to_mask
    upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
    # Random pixel patch x-coordinates.
    pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                  (num_patches, 1, 1))
    # Random pixel patch y-coordinates.
    pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                  (num_patches, 1, 1))
    # Add patch coordinate offsets.
    # Shape will broadcast to (num_patches, _patch_size, _patch_size).
    patch_dx_int, patch_dy_int = camera_utils.dilated_pixel_coordinates(
        self._patch_size, self._patch_size, self.dilate)
    pix_x_int = pix_x_int + patch_dx_int
    pix_y_int = pix_y_int + patch_dy_int
    # Random camera indices.
    if self._batching == utils.BatchingMethod.ALL_IMAGES:
      cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
    else:
      cam_idx = np.random.randint(0, self._n_examples, (1,))

    if self._apply_bayer_mask:
      # Compute the Bayer mosaic mask for each pixel in the batch.
      lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
    else:
      lossmult = None

    return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                lossmult=lossmult)

  def generate_ray_batch(self, cam_idx: int) -> utils.Batch:
    """Generate ray batch for a specified camera in the dataset."""
    if self._render_spherical:
      camtoworld = self.camtoworlds[cam_idx]
      rays = camera_utils.cast_spherical_rays(
          camtoworld, self.height, self.width, self.near, self.far, xnp=np)
      return utils.Batch(rays=rays)
    else:
      # Generate rays for all pixels in the image.
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.width, self.height)
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

  def _next_test(self) -> utils.Batch:
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples
    return self.generate_ray_batch(cam_idx)


class Onthego(Dataset):
  """on-the-go Dataset."""

  def load_feat(self, path, feat_rate, factor):
    image_dir = f'images_{factor}' if f'images_{factor}' in path else 'images'
    format = path[-4:]
    feat_path = path.replace(image_dir, f'features_{feat_rate}').replace(format, '.npy')
    feat = np.load(feat_path)
    return feat

  def _load_renderings(self, config):
    """Load images from disk."""
    # Set up scaling factor.
    image_dir_suffix = ''
    # Use downsampling factor (unless loading training split for raw dataset,

    image_dir_suffix = f'_{config.factor}'
    factor = config.factor
    pose_data = load_blender_posedata(self.data_dir)
    image_names, poses, pixtocam, distortion_params, camtype = pose_data

    # Previous NeRF results were generated with images sorted by filename,
    # use this flag to ensure metrics are reported on the same test set.
    if config.load_alphabetical:
      inds = np.argsort(image_names)
      image_names = [image_names[i] for i in inds]
      poses = poses[inds]

    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocam = pixtocam @ np.diag([factor, factor, 1.])
    self.pixtocams = pixtocam.astype(np.float32)
    self.focal = 1. / self.pixtocams[0, 0]
    self.distortion_params = distortion_params
    self.camtype = camtype

    raw_testscene = False
    if config.rawnerf_mode:
      # Load raw images and metadata.
      images, metadata, raw_testscene = raw_utils.load_raw_dataset(
          self.split,
          self.data_dir,
          image_names,
          config.exposure_percentile,
          factor)
      self.metadata = metadata

    else:
      # Load images.
      colmap_image_dir = os.path.join(self.data_dir, 'images')
      image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix)
      for d in [image_dir, colmap_image_dir]:
        if not utils.file_exists(d):
          raise ValueError(f'Image folder {d} does not exist.')
      # Downsampled images may have different names vs images used for COLMAP,
      # so we need to map between the two sorted lists of files.
      colmap_files = sorted(utils.listdir(colmap_image_dir))
      image_files = sorted(utils.listdir(image_dir))
      colmap_to_image = dict(zip(colmap_files, image_files))
      image_paths = [os.path.join(image_dir, colmap_to_image[f])
                     for f in image_names]
      images = [utils.load_img(x) for x in image_paths]
      images = np.stack(images, axis=0) / 255.

      # read in features
      features = [self.load_feat(x, config.feat_rate, config.factor) for x in image_paths]
      features = np.stack(features, axis=0)

      # create assignment
      # assignment is for super-pixel setting, not used in this project
      patch_H, patch_W = config.H // config.feat_rate, config.W //config.feat_rate
      patch_H = patch_H // config.feat_ds * config.feat_ds
      patch_W = patch_W // config.feat_ds * config.feat_ds
      i, j = np.meshgrid(np.arange(patch_H), np.arange(patch_W), indexing='ij')
      assignment_data = i // config.feat_ds * patch_W // config.feat_ds + j // config.feat_ds
      assignment_data_resized = cv2.resize(assignment_data, 
                                           (images.shape[2], images.shape[1]), 
                                           interpolation=cv2.INTER_NEAREST).astype(np.int64)
      assignments = assignment_data_resized[None,...].repeat(features.shape[0], axis=0)
      # EXIF data is usually only present in the original JPEG images.
      jpeg_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
      exifs = [utils.load_exif(x) for x in jpeg_paths]
      self.exifs = exifs
      if 'ExposureTime' in exifs[0] and 'ISOSpeedRatings' in exifs[0]:
        gather_exif_value = lambda k: np.array([float(x[k]) for x in exifs])
        shutters = gather_exif_value('ExposureTime')
        isos = gather_exif_value('ISOSpeedRatings')
        self.exposures = shutters * isos / 1000.

    # Load bounds if possible (only used in forward facing scenes).
    posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
    with open(os.path.join(self.data_dir, 'split.json'), 'r') as file:
      split_json = json.load(file)
    if utils.file_exists(posefile):
      with utils.open_file(posefile, 'rb') as fp:
        poses_arr = np.load(fp)
      bounds = poses_arr[:, -2:]
    else:
      bounds = np.array([0.01, 1.])
    self.colmap_to_world_transform = np.eye(4)

    # Separate out 360 versus forward facing scenes.
    if config.forward_facing:
      # Set the projective matrix defining the NDC transformation.
      self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
      # Rescale according to a default bd factor.
      scale = 1. / (bounds.min() * .75)
      poses[:, :3, 3] *= scale
      self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
      bounds *= scale
      # Recenter poses.
      poses, transform = camera_utils.recenter_poses(poses)
      self.colmap_to_world_transform = (
          transform @ self.colmap_to_world_transform)
      # Forward-facing spiral render path.
      self.render_poses = camera_utils.generate_spiral_path(
          poses, bounds, n_frames=config.render_path_frames)
    else:
      # Rotate/scale poses to align ground with xy plane and fit to unit cube.
      poses, transform = camera_utils.transform_poses_pca(poses)
      self.colmap_to_world_transform = transform
      if config.render_spline_keyframes is not None:
        rets = camera_utils.create_render_spline_path(config, image_names,
                                                      poses, self.exposures)
        self.spline_indices, self.render_poses, self.render_exposures = rets
      else:
        # Automatically generated inward-facing elliptical render path.
        self.render_poses = camera_utils.generate_ellipse_path(
            poses,
            n_frames=config.render_path_frames,
            z_variation=config.z_variation,
            z_phase=config.z_phase)
    if raw_testscene:
      # For raw testscene, the first image sent to COLMAP has the same pose as
      # the ground truth test image. The remaining images form the training set.
      raw_testscene_poses = {
          utils.DataSplit.TEST: poses[:1],
          utils.DataSplit.TRAIN: poses[1:],
      }
      poses = raw_testscene_poses[self.split]

    self.poses = poses

    # Select the split according to the split file.
    all_indices = np.arange(images.shape[0])
    split_indices = {}
    if config.eval_train:
      split_indices[utils.DataSplit.TEST] = all_indices[split_json['clutter']]
    else:
      split_indices[utils.DataSplit.TEST] = all_indices[split_json['extra']]
    if config.train_clean:
      split_indices[utils.DataSplit.TRAIN] = all_indices[split_json['clean']]
    else:
      split_indices[utils.DataSplit.TRAIN] = all_indices[split_json['clutter']]

    indices = split_indices[self.split]
    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    poses = poses[indices]
    features = features[indices]
    assignments = assignments[indices]
    if self.exposures is not None:
      self.exposures = self.exposures[indices]
    if config.rawnerf_mode:
      for key in ['exposure_idx', 'exposure_values']:
        self.metadata[key] = self.metadata[key][indices]

    self.images = images
    self.camtoworlds = self.render_poses if config.render_path else poses
    self.height, self.width = images.shape[1:3]

    self.features = features
    self.assignments = assignments