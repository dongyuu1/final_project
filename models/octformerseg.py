# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
from ocnn.octree import Octree
from typing import List, Dict
from .octformer import OctFormer


class SegHeader(torch.nn.Module):

  def __init__(
          self, out_channels: int, channels: List[int], fpn_channel: int,
          nempty: bool, num_up: int = 1, dropout: List[float] = [0.0, 0.0]):
    super().__init__()
    self.num_up = num_up
    channels = channels[2:]
    self.num_stages = len(channels)
    self.conv1x1 = torch.nn.ModuleList([torch.nn.Linear(
        channels[i], fpn_channel) for i in range(self.num_stages-1, -1, -1)])
    self.upsample = ocnn.nn.OctreeUpsample('nearest', nempty)
    self.conv3x3 = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
        fpn_channel, fpn_channel, kernel_size=[3],
        stride=1, nempty=nempty) for i in range(self.num_stages)])
    self.up_conv = torch.nn.ModuleList([ocnn.modules.OctreeDeconvBnRelu(
        fpn_channel, fpn_channel, kernel_size=[3],
        stride=2, nempty=nempty) for i in range(self.num_up)])
    self.interp = ocnn.nn.OctreeInterp('nearest', nempty)
    self.classifier = torch.nn.Sequential(
        torch.nn.Dropout(dropout[0]),
        torch.nn.Linear(fpn_channel, fpn_channel),
        torch.nn.BatchNorm1d(fpn_channel),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(dropout[1]),
        torch.nn.Linear(fpn_channel, out_channels),)

  def forward(self, features: Dict[int, torch.Tensor], octree: Octree,
              query_pts: torch.Tensor):
    #print(features)
    depth = min(features.keys())
    depth_max = max(features.keys())
    
    assert self.num_stages == len(features)

    feature = self.conv1x1[0](features[depth])
    conv_out = self.conv3x3[0](feature, octree, depth)
    #print(octree.nnum_nempty[depth], feature.shape, conv_out.shape)
    out = self.upsample(conv_out, octree, depth, depth_max)
    #out2 = self.upsample(conv_out, octree, depth, depth_max)
    #out = (out1 + out2) / 2
    #self.upsample1(conv_out, octree, depth, depth_max)
    #print(octree.nnum_nempty[depth+1], out.shape)
    for i in range(1, self.num_stages):
      depth_i = depth + i
      feature = self.upsample(feature, octree, depth_i - 1, depth_i) 
      #self.upsample1(feature, octree, depth_i-1, depth_i)
      #self.upsample1(conv_out, octree, depth_i-1, depth_max)
      #print(octree.nnum_nempty[depth_i-1], feature.shape)
      feature = self.conv1x1[i](features[depth_i]) + feature
      conv_out = self.conv3x3[i](feature, octree, depth_i)
      out = out + self.upsample(conv_out, octree, depth_i, depth_max) 

    for i in range(self.num_up):
      out = self.up_conv[i](out, octree, depth_max + i)  # upsample
    out = self.interp(out, octree, depth_max + self.num_up, query_pts)
    out = self.classifier(out)
    return out
  
  def upsample_parents(self, data: torch.Tensor, octree: Octree, depth: int, max_depth:int):
    assert depth <= max_depth
    if depth == max_depth:
      return data
      
    stage = max_depth - depth
    #pidx = torch.arange(0, octree.nnum_nempty[depth]) 
    pidx = octree.pidx[depth + 1]
  
    if stage > 1:   
      for i in range(stage - 1):
        depth_i = depth + i
        pidx = pidx[octree.pidx[depth_i+2]]
    
    return data[pidx]
        

class OctFormerSeg(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int,
          channels: List[int] = [96, 192, 384, 384],
          num_blocks: List[int] = [2, 2, 18, 2],
          num_heads: List[int] = [6, 12, 24, 24],
          patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
          nempty: bool = True, stem_down: int = 2, head_up: int = 2,
          fpn_channel: int = 168, head_drop: List[float] = [0.0, 0.0], **kwargs):
    super().__init__()
    self.backbone = OctFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
    self.head = SegHeader(
        out_channels, channels, fpn_channel, nempty, head_up, head_drop)
    self.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.trunc_normal_(m.weight, std=0.02)
      if isinstance(m, torch.nn.Linear) and m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int,
              query_pts: torch.Tensor, visual=False):
    features, _, _ = self.backbone(data, octree, depth, visual=visual)
    output = self.head(features, octree, query_pts)
    return output, _, _
