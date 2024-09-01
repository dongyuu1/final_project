# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import ocnn
from thsolver import Dataset
from ocnn.octree import Points
from typing import Union, List
from ocnn.octree.shuffled_key import xyz2key, key2xyz
from ocnn.utils import meshgrid, scatter_add, cumsum, trunc_div
from .utils import ReadPly
from .octree import Octree, merge_octrees


class TransformO:
  r''' A boilerplate class which transforms an input data for :obj:`ocnn`.
  The input data is first converted to :class:`Points`, then randomly transformed
  (if enabled), and converted to an :class:`Octree`.

  Args:
    depth (int): The octree depth.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.
    distort (bool): If true, performs the data augmentation.
    angle (list): A list of 3 float values to generate random rotation angles.
    interval (list): A list of 3 float values to represent the interval of
        rotation angles.
    scale (float): The maximum relative scale factor.
    uniform (bool): If true, performs uniform scaling.
    jittor (float): The maximum jitter values.
    orient_normal (str): Orient point normals along the specified axis, which is
        useful when normals are not oriented.
  '''

  def __init__(self, depth: int, full_depth: int, distort: bool, angle: list,
               interval: list, scale: float, uniform: bool, jitter: float,
               flip: list, orient_normal: str = '', **kwargs):
    super().__init__()

    # for octree building
    self.depth = depth
    self.full_depth = full_depth

    # for data augmentation
    self.distort = distort
    self.angle = angle
    self.interval = interval
    self.scale = scale
    self.uniform = uniform
    self.jitter = jitter
    self.flip = flip

    # for other transformations
    self.orient_normal = orient_normal

  def __call__(self, sample: dict, idx: int):
    r''''''

    points = self.preprocess(sample, idx)
    output = self.transform(points, idx)
    output['octree'] = self.points2octree(output['points'])
    return output

  def preprocess(self, sample: dict, idx: int):
    r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
    transformations, like normalization.
    '''

    xyz = torch.from_numpy(sample['points'])
    normals = torch.from_numpy(sample['normals'])
    points = Points(xyz, normals)
    return points

  def transform(self, points: Points, idx: int):
    r''' Applies the general transformations provided by :obj:`ocnn`.
    '''

    # The augmentations including rotation, scaling, and jittering.
    if self.distort:
      rng_angle, rng_scale, rng_jitter, rnd_flip = self.rnd_parameters()
      points.flip(rnd_flip)
      points.rotate(rng_angle)
      points.translate(rng_jitter)
      points.scale(rng_scale)

    if self.orient_normal:
      points.orient_normal(self.orient_normal)

    # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
    inbox_mask = points.clip(min=-1, max=1)
    return {'points': points, 'inbox_mask': inbox_mask}

  def points2octree(self, points: Points):
    r''' Converts the input :attr:`points` to an octree.
    '''

    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def rnd_parameters(self):
    r''' Generates random parameters for data augmentation.
    '''

    rnd_angle = [None] * 3
    for i in range(3):
      rot_num = self.angle[i] // self.interval[i]
      rnd = torch.randint(low=-rot_num, high=rot_num+1, size=(1,))
      rnd_angle[i] = rnd * self.interval[i] * (3.14159265 / 180.0)
    rnd_angle = torch.cat(rnd_angle)

    rnd_scale = torch.rand(3) * (2 * self.scale) - self.scale + 1.0
    if self.uniform:
      rnd_scale[1] = rnd_scale[0]
      rnd_scale[2] = rnd_scale[0]

    rnd_flip = ''
    for i, c in enumerate('xyz'):
      if torch.rand([1]) < self.flip[i]:
        rnd_flip = rnd_flip + c

    rnd_jitter = torch.rand(3) * (2 * self.jitter) - self.jitter
    return rnd_angle, rnd_scale, rnd_jitter, rnd_flip

class Transform(TransformO):
  r''' Wraps :class:`ocnn.data.Transform` for convenience.
  '''

  def __init__(self, flags):
    super().__init__(**flags)
    self.flags = flags

class CollateBatch:
  r''' Merge a list of octrees and points into a batch.
  '''

  def __init__(self, merge_points: bool = False):
    self.merge_points = merge_points

  def __call__(self, batch: list):
    assert type(batch) == list

    outputs = {}
    for key in batch[0].keys():
      outputs[key] = [b[key] for b in batch]

      # Merge a batch of octrees into one super octree
      if 'octree' in key:
        octree = merge_octrees(outputs[key])
        # NOTE: remember to construct the neighbor indices
        octree.construct_all_neigh()
        outputs[key] = octree

      # Merge a batch of points
      if 'points' in key and self.merge_points:
        outputs[key] = ocnn.octree.merge_points(outputs[key])

      # Convert the labels to a Tensor
      if 'label' in key:
        outputs['label'] = torch.tensor(outputs[key])
    octree = outputs["octree"]
    return outputs


class ModelNetTransform(Transform):

  def preprocess(self, sample: dict, idx: int):
    points = super().preprocess(sample, idx)

    # Comment out the following lines if the shapes have been normalized
    # in the dataset generation stage.
    #
    # Normalize the points into one unit sphere in [-0.8, 0.8]
    # bbmin, bbmax = points.bbox()
    # points.normalize(bbmin, bbmax, scale=0.8)
    #
    # points.scale(torch.Tensor([0.8, 0.8, 0.8]))

    return points


def read_file(filename: str):
  filename = filename.replace('\\', '/')
  if filename.endswith('.ply'):
    read_ply = ReadPly(has_normal=True)
    return read_ply(filename)
  elif filename.endswith('.npz'):
    raw = np.load(filename)
    output = {'points': raw['points'], 'normals': raw['normals']}
    return output
  else:
    raise ValueError


def get_modelnet40_dataset(flags):
  transform = ModelNetTransform(flags)
  collate_batch = CollateBatch()

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, take=flags.take)
  return dataset, collate_batch
