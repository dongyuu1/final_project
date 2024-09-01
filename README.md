# 3D Object Recognition and Classification using Artificial Intelligence
This is the code for the final project of Dongyu Wang, student number 23104424. The implementation 
is mainly based on the official code of Octformer (https://github.com/octree-nn/octformer?tab=readme-ov-file) with several modifications.


The new content added includes:

1. Rewrite the octree structure to make it compatible with Octree Spherical Convolutional Neural Networks. 
The new code is placed at ./models/octree.py. The changes include: modify Octree.reset(),
modify Octree.build_octree(), modify Octree.octree_grow_full(), modify Octree.to(), and modify merge_octrees().
2. Change the architecture of backbone. The code is at ./models/octformer.py. The changes include: add 
a new class OctSphericalCNN (corresponds to a Attn+CNN block mentioned in the paper), add a new class 
SphericalConv (corresponds to a Convolutional Phase layer), and modify Octformer.forward(). 
3. Change the architecture of classification head and segmentation head. The code is at ./models/octformercls and 
./models/octformerseg. The changes include: modify OctFormerCls.\_\_init__(), modify 
OctFormerCls.forward(), modify OctFormerSeg.forward(), modify SegHeader.forward(). 
4. Write code to visualise intermediate features, including ./classification_visual.py, 
./segmentation_visual.py, present_cls.py, present_seg.py, and plot_training_curve.py. 


## 1. Installation


1. Run the following command to intall pytorch 1.12.1.

    ```bash
    conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

2. Install the required packages.

    ```bash
    pip install -r requirements.txt
    ```

3. Install the depth-wise convolution package.

    ```bash
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```

## 2. ModelNet40 Classification

1. Run the following command to prepare the dataset.

    ```bash
    python tools/cls_modelnet.py
    ```
2. Run the following command to train the network with 1 GPU. 
    ```bash
    python classification.py --config configs/cls_m40.yaml SOLVER.gpu 0,
    ```
## 3. ScanNet Segmentation

1. **Data**: Download the data from https://kaldir.vc.in.tum.de/scannet_benchmark/.
   Unzip the data and place it to the folder ./data. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in <scannet_folder>
    ```

2. **Train**: Run the following command to train the network with 2 GPUs and
   port 10002.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0,1 --alias scannet --port 10002
    ```

## 4. Visualisation

1. Run the following code to output visualisation data for point cloud classification.

    ```bash
    python classification_visual.py --config configs/cls_m40_test.yaml SOLVER.gpu 0,
    ```
2. Download well-trained model weights for point cloud segmentation via the link:
   https://drive.google.com/file/d/1mQrGqGzXJ4yQYuruo-tIAKlkY74wSgY4/view?usp=sharing. Place them to the root path. 

3. Run the following code to output visualisation data for point cloud segmentation. 

    ```bash
    python scripts/run_seg_scannet_test.py --gpu 0 --alias scannet --port 10002
    ```
4. Run the following code to visualise features from point cloud classification.
    ```bash
    python present_cls.py
    ```
5. Run the following code to visualise segmentation results. 
    ```bash
    python present_seg.py
    ```
6. Run the following code to plot curves for different metrics during training.
    ```bash
    python plot_training_curve.py
    ```
## 5. Citation

   ```bibtex
    @article {OctFormer,
        title      = {OctFormer: Octree-based Transformers for {3D} Point Clouds},
        author     = {Wang, Peng-Shuai},
        journal    = {ACM Transactions on Graphics (SIGGRAPH)},
        volume     = {42},
        number     = {4},
        year       = {2023},
    }
   ```
   ```bibtex
    @InProceedings{Phai-CNN,
        author = {Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
        title = {Octree Guided CNN With Spherical Kernels for 3D Point Clouds},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        pages = {9623-9632},
        month = {Long Beach, USA},
        year = {2019}
    }
   ```
