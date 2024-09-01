# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import ocnn
import os
from tqdm import tqdm
from thsolver import Solver
from datasets import get_modelnet40_dataset
from builder import get_classification_model
from thsolver.config import parse_args
import open3d
import numpy as np


class ClsSolver(Solver):

    def get_model(self, flags):
        return get_classification_model(flags)

    def get_dataset(self, flags):
        return get_modelnet40_dataset(flags)

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    def forward(self, batch):
        octree, label = batch['octree'].cuda(), batch['label'].cuda()

        data = self.get_input_feature(octree)
        logits = self.model(data, octree, octree.depth)
        log_softmax = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_softmax, label)
        pred = torch.argmax(logits, dim=1)
        accu = pred.eq(label).float().mean()
        # pred_onehot = torch.nn.functional.one_hot(pred, flags.nout)

        # label_onehot = torch.nn.functional.one_hot(label, flags.nout)
        # cls_count = torch.sum(label_onehot, dim=0)
        # emp_mask = (label_onehot == 0)
        # label_onehot[emp_mask] = -1

        # correct = torch.sum(label_onehot.eq(pred_onehot), dim=0)
        # print("cls", cls_count)
        # print("cor", correct)
        return loss, accu

    def train_step(self, batch):
        loss, accu = self.forward(batch)
        return {'train/loss': loss, 'train/accu': accu}

    def test_step(self, batch):
        with torch.no_grad():
            loss, accu = self.forward(batch)
        return {'test/loss': loss, 'test/accu': accu}


def to_points(octree, depth, rescale: bool = True):
    # by default, use the average points generated when building the octree
    # from the input point cloud
    xyz = octree.points[depth]

    # xyz is None when the octree is predicted by a neural network
    if xyz is None:
        x, y, z, _ = octree.xyzb(depth, nempty=False)
        xyz = torch.stack([x, y, z], dim=1) + 0.5

    return xyz


def main():
    flags = parse_args()
    model = get_classification_model(flags.MODEL).cuda()
    model.load_state_dict(torch.load("./logs/m40/octformer_d6/checkpoints/00200.model.pth"))
    dataset, collate_fn = get_modelnet40_dataset(flags.DATA.test)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=flags.DATA.test.batch_size, collate_fn=collate_fn, pin_memory=flags.DATA.test.pin_memory)
    test_iter = iter(dataloader)
    eval_step = len(dataloader)

    for i in tqdm(range(1), ncols=80, leave=False):
        batch = next(test_iter)
        with torch.no_grad():
            octree, label = batch['octree'].cuda(), batch['label'].cuda()
            sub_flag = flags.MODEL
            octree_feature = ocnn.modules.InputFeature(sub_flag.feature, sub_flag.nempty)
            data = octree_feature(octree)
            logits, inter_attn_feat_list, cnn_inter_feat_list = model(data, octree, octree.depth, visual=True)
            stage = 1
            block = 1
            D = 4
            # attn_feat_visual_generation(inter_attn_feat_list, stage, block, D, octree)
            cnn_feat_visual_generation(cnn_inter_feat_list, stage)


def cnn_feat_visual_generation(inter_feat_list, stage):
   inter_feat_tuple = inter_feat_list[stage]
   cnn_feats, pidx = inter_feat_tuple
  
   N = cnn_feats.shape[1]
   cnn_feats, pidx = inter_feat_list[stage]
   #print(cnn_feats.shape, torch.max(pidx))

   points = cnn_feats[0]
   central_points = cnn_feats[1]
   total_points = torch.cat([points, central_points], dim=0).cpu().numpy()

   line_order = torch.linspace(0, N-1, N, dtype=torch.int32)
   print(line_order)
   pidx = pidx.cpu()
   lines = torch.stack([line_order, line_order + N], dim=-1).numpy()

   line_set = open3d.geometry.LineSet(points=open3d.utility.Vector3dVector(total_points),
                                      lines=open3d.utility.Vector2iVector(lines))
    
   point_cloud = open3d.geometry.PointCloud()
   point_cloud.points = open3d.utility.Vector3dVector(total_points)
   colors = np.zeros((N * 2, 3))
   colors[:N, 2] = 1
   colors[N:, 0] = 1
   point_cloud.colors = open3d.utility.Vector3dVector(colors)
   open3d.io.write_point_cloud("./points.pcd", point_cloud)
    
   open3d.io.write_line_set("./lineset.ply", line_set)


def attn_feat_visual_generation(inter_feat_list, stage, block, D, octree):
    N, H, P, _ = inter_feat_list[stage][block].shape

    attn_feature = inter_feat_list[stage][block].transpose(1, 2).reshape(-1, H, P)

    init_xyz = to_points(octree, 2+stage)
    pad_num = N * P - init_xyz.shape[0]
    pad_t = torch.zeros((pad_num, 3), device="cuda:0")
    xyz = torch.cat([init_xyz, pad_t])

    xyz = xyz.view(-1, P, D, 3).transpose(1, 2).reshape(-1, P, 3)

    print(xyz[0, :, :].shape, attn_feature[0, 0, :].unsqueeze(-1).shape)
    scale = (attn_feature[0, 0, :] / torch.max(attn_feature[0, 0, :])).cpu().numpy()
    colors = np.zeros((P, 3))
    colors[:, 0] = scale
    colors[:, 2] = 1 - scale

    # xyz_part = xyz[0, :, :] + attn_feature[0, 0, :]
    init_xyz = init_xyz.cpu().numpy()

    point_cloud_attn = open3d.geometry.PointCloud()
    point_cloud_attn.points = open3d.utility.Vector3dVector(xyz[0, :, :].cpu().numpy())
    point_cloud_attn.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud("./attn.pcd", point_cloud_attn)

    colors = (np.ones((N, P))[:, :, None] * np.random.randint(low=0, high=256, size=(N, 1, 3)))
    #colors[4:, :, :] = np.ones((N - 4, P, 3)) * 255
    colors = colors.reshape(N * P, 3) / 255

    point_cloud_patches = open3d.geometry.PointCloud()
    point_cloud_patches.points = open3d.utility.Vector3dVector(xyz.view(-1, 3).cpu().numpy())
    point_cloud_patches.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud("./patches.pcd", point_cloud_patches)
    # print(len(keys))
    
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyz.view(-1, 3).cpu().numpy())
    colors = np.zeros((N * P, 3)) + 0.5
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud("./point_cloud.pcd", point_cloud)

if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
