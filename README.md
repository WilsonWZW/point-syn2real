# Point-Syn2Real
This is the official implementation of our ICME 2023 Paper:
[**Point-Syn2Real: Semi-Supervised Synthetic-to-Real Cross-Domain Learning for Object Classification in 3D Point Clouds**](https://arxiv.org/pdf/2210.17009.pdf)

Example data available at: [Google-Drive](https://drive.google.com/drive/folders/1yhbtdf8DqJnrLDLv3IHSshoxuFYqAXcw?usp=sharing)

# Overview
Object classification using LiDAR 3D point cloud data is critical for modern applications such as autonomous driving. However, labeling point cloud data is labor-intensive as it requires human annotators to visualize and inspect the 3D data from different perspectives. In this paper, we propose a semi-supervised cross-domain learning approach that does not rely on manual annotations of point clouds and performs similar to fully-supervised approaches. We utilize available 3D object models to train classifiers that can generalize to real-world point clouds. We simulate the acquisition of point clouds by sampling 3D object models from multiple viewpoints and with arbitrary partial occlusions. We then augment the resulting set of point clouds through random rotations and adding Gaussian noise to better emulate the real-world scenarios. We then train point cloud encoding models on the synthesized and augmented datasets and evaluate their cross-domain classification performance on corresponding real-world datasets. We also introduce Point-Syn2Real, a new benchmark dataset for cross-domain learning on point clouds. The results of our extensive experiments with this dataset demonstrate that the proposed cross-domain learning approach for point clouds outperforms the related baseline and state-of-the-art approaches in both indoor and outdoor settings in terms of cross-domain generalizability.

# Instruction
## Dataset
Download our prepared example dataset [here](https://drive.google.com/drive/folders/1yhbtdf8DqJnrLDLv3IHSshoxuFYqAXcw?usp=sharing). Please extract your dataset in the data/ folder.

Our 3D dataset is collected from 3D datasets including [SemanticKitti](http://www.semantic-kitti.org/), [ModelNet](https://modelnet.cs.princeton.edu/), [ShapeNet](https://shapenet.cs.stanford.edu/iccv17/), [ScanNet](http://www.scan-net.org/), and publically available 3D CAD models from [3D-Warehouse](https://3dwarehouse.sketchup.com/).

The multiview simulation dataset is prepared using Blender. If you are generating new multi-view point cloud dataset, please follow the instruction in OcCo project: [OcCo](https://github.com/hansen7/OcCo/tree/master/render).


# Citation
Please consider to cite this work if it is helpful.
```bibtex
@inproceedings{DBLP:conf/icmcs/WangALBBH23,
  author       = {Ziwei Wang and
                  Reza Arablouei and
                  Jiajun Liu and
                  Paulo Borges and
                  Greg Bishop{-}Hurley and
                  Nicholas Heaney},
  title        = {Point-Syn2Real: Semi-Supervised Synthetic-to-Real Cross-Domain Learning
                  for Object Classification in 3D Point Clouds},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2023,
                  Brisbane, Australia, July 10-14, 2023},
  pages        = {1481--1486},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/ICME55011.2023.00256},
  doi          = {10.1109/ICME55011.2023.00256},
}
```

# Acknowledgement
Thanks to the following repo:
[Blender]()
[PointDAN](https://github.com/canqin001/PointDAN)
[OcCo](https://github.com/hansen7/OcCo/tree/master/render)
