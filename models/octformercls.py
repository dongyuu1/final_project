import ocnn
import torch
from ocnn.octree import Octree
from typing import List
from .octformer import OctFormer


class ClsHeader(torch.nn.Module):
  def __init__(self, out_channels: int, in_channels: int,
               nempty: bool = False, dropout: float = 0.5):
    super().__init__()
    self.cls_header = torch.nn.Sequential(
        ocnn.modules.FcBnRelu(in_channels, 256),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(256, out_channels))

  def forward(self, data: torch.Tensor):
    logit = self.cls_header(data)
    return logit


class OctFormerCls(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               channels: List[int] = [96, 192, 384, 384],
               num_blocks: List[int] = [2, 2, 18, 2],
               num_heads: List[int] = [6, 12, 24, 24],
               patch_size: int = 32, dilation: int = 4,
               drop_path: float = 0.5, nempty: bool = True,
               stem_down: int = 2, head_drop: float = 0.5, **kwargs):
    super().__init__()
    self.backbone = OctFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
    in_channels = 0
    for i in channels[stem_down:]:
      in_channels += i
    self.head = ClsHeader(
        out_channels, in_channels, nempty, head_drop)
    self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
    self.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.trunc_normal_(m.weight, std=0.02)
      if isinstance(m, torch.nn.Linear) and m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int, visual=False):
    features, attn_list, cnn_inter_feat_list = self.backbone(data, octree, depth, visual)
    min_depth = min(features.keys())
    max_depth = max(features.keys())
    
    feature_list = []
    for depth in range(min_depth, max_depth+1):

      feature = self.global_pool(features[depth], octree, depth)
      feature_list.append(feature)
    features = torch.cat(feature_list, dim=1)
    
    output = self.head(features)
    return output, attn_list, cnn_inter_feat_list
