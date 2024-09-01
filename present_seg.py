# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import open3d as o3d
import numpy as np


def main():
  visualize_seg(input="parent")
  visualize_seg(input="nearest")
  visualize_seg(input="labels")

def visualize_seg(input: str):
  pcd = o3d.io.read_point_cloud('./visualisation/segmentation_{}.pcd'.format(input))
  R = pcd.get_rotation_matrix_from_xyz((np.pi / 180 * (-70), 0, 270 * np.pi / 180))
  pcd.rotate(R, center=(0, 0, 0))
  o3d.visualization.draw_geometries([pcd])

  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(pcd)
  vis.destroy_window()

if __name__ == "__main__":
  #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
  main()
