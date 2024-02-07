#!/bin/bash
python train_cls.py -s modelnet_multiview -t scannet --result_append new_modelnet_full --epoch 80 --core_model dgcnn --da_method entropy --da_loss_alpha 0.2