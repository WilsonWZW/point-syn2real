import os
import time
import json
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, accuracy_score

import torch
import torch.nn as nn
import torch.multiprocessing
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from utils.custom_datasets import *
from utils.helper import *
from utils.adv_loss import AdversarialLoss
from models.cls_models import DGCNNNet, PointNet2

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(0)
np.random.seed(0)

# Command settings
parser = argparse.ArgumentParser(description='Syn2Real Object Detection')
parser.add_argument('--source','-s', type=str, 
    choices=['modelnet_sub', 'modelnet_multiview', 'shapenet_sub', 'shapenet_multiview', 'scannet', '3d_city', '3d_city_multiview', 'semkitti_sub'],
    help='source dataset, same as dataset folder name', 
    default='modelnet_sub')
parser.add_argument('--target','-t', type=str, 
    choices=['modelnet_sub', 'modelnet_multiview', 'shapenet_sub', 'shapenet_multiview', 'scannet', '3d_city', '3d_city_multiview', 'semkitti_sub'],
    help='target dataset, same as dataset folder name.', 
    default='scannet')
parser.add_argument('--data_root', type=str, 
    help='Root folder of dataset', 
    default='/home/user1/datastore/Datasets/PointDA_data')
parser.add_argument('--result_root', type=str, 
    help='Root folder of result', 
    default='/home/user1/datastore/Datasets/PointDA_data/results')
parser.add_argument('--result_append', type=str, 
    help='Note for the saved result', 
    default='multiview')
parser.add_argument('--flag_weighted_loss', type=str,
    choices=['None', 'source_set', 'target_set'],
    help='Class weights for imbalanced dataset. opt: None | source_set | target_set', 
    default='source_set')
parser.add_argument('--build_graph', 
    help='Flag for whether build a graph based on points',
    action='store_true',
    default=False)
parser.add_argument('--no_random_aug',
    help='Flag for whether random augment the points',
    action='store_true',
    default=False)
parser.add_argument('--learning_rate', type=float,
    help='Initial learning rate 1e-3 | 1e-4',
    default=1e-3)
parser.add_argument('--num_point', type=int,
    help='Number of points for each cloud',
    default=1024)
parser.add_argument('--epoch', type=int,
    help='Number of epochs',
    default=80)
parser.add_argument('--core_model', type=str,
    choices=['dgcnn','dgin','gin','pointnet','dgraph'],
    help='the point encoder network',
    default='dgcnn')
parser.add_argument('--da_method', type=str,
    choices=['none','entropy','mmd','dann'],
    help='the domain adaptation method',
    default='none')
parser.add_argument('--da_loss_alpha', type=float,
    help='the domain adaptation loss weight',
    default=1)
parser.add_argument('--xe_loss_weight', type=float,
    help='the cross-entropy loss weight',
    default=1)

args = parser.parse_args()
print(args)

# Source
if args.source=='3d_city':
    s_pre_transform = T.Compose([T.NormalizeScale(), T.SamplePoints(4096)])
elif args.source=='3d_city_multiview':
    s_pre_transform = T.Compose([T.NormalizeScale(), T.FixedPoints(4096)])
else:
    s_pre_transform = T.Compose([T.NormalizeScale()])

if args.build_graph:
    # with Graph. KNN graph k=4, or Radius r=0.2
    # 360 Random rotation on z-axis
    s_transform = T.Compose([T.FixedPoints(args.num_point), T.KNNGraph(k=4),
        T.RandomRotate(360, axis=2),
        RandomJitter(sigma=0.01, clip=0.02)])
elif args.no_random_aug:
    # no graph, no augmentation
    s_transform = T.Compose([T.FixedPoints(args.num_point)])
else: 
    # for Non-graph based
    # 360 Random rotation on z-axis
    s_transform = T.Compose([T.FixedPoints(args.num_point),
        T.RandomRotate(360, axis=2),
        RandomJitter(sigma=0.01, clip=0.02)])

# Target
if args.target=='3d_city':
    t_pre_transform = T.Compose([T.NormalizeScale(), T.SamplePoints(4096)])
elif args.target=='3d_city_multiview':
    t_pre_transform = T.Compose([T.NormalizeScale(), T.FixedPoints(4096)])
else:
    t_pre_transform = T.Compose([T.NormalizeScale()])

if args.build_graph: 
    # with Graph. KNN graph k=4, or Radius r=0.2
    t_transform = T.Compose([T.FixedPoints(args.num_point), T.KNNGraph(k=4)])
else: # for Non-graph based
    t_transform = T.Compose([T.FixedPoints(args.num_point)])

# Dataset args to Dataset Class
dataset_func = {'modelnet_sub': ModelNet_DA_Subset, 
    'modelnet_multiview': ModelNet_Multiview, 
    'shapenet_sub': ShapeNet_DA_Subset, 
    'shapenet_multiview': ShapeNet_Multiview,
    'scannet': Scannet_ObjectDB,
    '3d_city': Simu3D_City, 
    '3d_city_multiview': Simu3D_City_Multiview, 
    'semkitti_sub': SemKitti_Object
    }


source_path = os.path.join(args.data_root, args.source)
target_path = os.path.join(args.data_root, args.target)
train_dataset = dataset_func[args.source](root=source_path, train=True, pre_transform=s_pre_transform, transform=s_transform)
test_dataset = dataset_func[args.target](root=target_path, train=False, pre_transform=t_pre_transform, transform=t_transform)
# For calculating the class weight ONLY
target_train_full = dataset_func[args.target](root=target_path, train=True, pre_transform=t_pre_transform, transform=t_transform)


val_rate = 0.2
val_len = int(val_rate * len(target_train_full))
val_len = min(val_len, len(test_dataset))
train_len = len(target_train_full) - val_len
target_train_subset, target_val_subset = random_split(target_train_full, [train_len,val_len], 
    generator=torch.Generator().manual_seed(0))

# Check if there is empty data point (for dataset bug in multiview)
try:
    for idx, item in enumerate(train_dataset):
        assert torch.sum(item.pos).isnan() == False, "Dataset AssertionError: Found NaN in Train dataset"
    for idx, item in enumerate(target_train_full):
        assert torch.sum(item.pos).isnan() == False, "Dataset AssertionError: Found NaN in Train dataset"
    for idx, item in enumerate(test_dataset):
        assert torch.sum(item.pos).isnan() == False, "Dataset AssertionError: Found NaN in Test dataset"
except AssertionError as msg:
    print (msg)
    print (idx, item.pos)
    exit(1)

# Save directory
timestr = time.strftime("%Y%m%d_%H%M%S")
print (timestr)
result_name = timestr + '-' + args.source + '-' + args.target + '-' + args.result_append
result_dir = os.path.join(args.result_root, result_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def main():
    
    print ('Report')
    print ('Source train:',source_path)
    print (s_pre_transform, s_transform)
    print ('Target Test:', target_path)
    print (t_pre_transform, t_transform)

    print ('Source - Train set')
    source_train_percent = stats_print(train_dataset)
    print ('Target - Train set')
    target_train_percent = stats_print(target_train_full)
    print ('Target - Test (Display only. Not used in training)')
    target_test_percent = stats_print(test_dataset)

    loss_class_weights = torch.Tensor(get_class_weight(source_train_percent, target_train_percent, args.flag_weighted_loss))
    
    s_train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6)
    t_train_loader = DataLoader(
        target_train_subset, batch_size=32, shuffle=True, num_workers=6)
    t_test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=6)
    t_val_loader = DataLoader(
        target_val_subset, batch_size=32, shuffle=False, num_workers=6)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.core_model == 'dgcnn':
        model = DGCNNNet(out_channels=train_dataset.num_classes, k=20).to(device)
    elif args.core_model == 'dgin':
        model = DynamicGIN_Net(in_channels=3, dim=64, out_channels=train_dataset.num_classes, k=20).to(device)
    elif args.core_model == 'gin':
        model = GIN_Net(in_channels=3, dim=512, out_channels=train_dataset.num_classes).to(device)
    elif args.core_model == 'dgraph':
        model = DynamicGraph_Conv_Net(in_channels=3, dim=64, out_channels=train_dataset.num_classes, k=20).to(device)
    elif args.core_model == 'pointnet':
        model = PointNet2(out_channels=train_dataset.num_classes).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    log_softmax = nn.LogSoftmax(dim=-1)
    nll_loss_func = nn.NLLLoss(weight=loss_class_weights.to(device), reduction='mean')

    if args.da_method == 'entropy':
        da_loss_func = EntropyLoss(reduction='mean')
    elif args.da_method == 'mmd':
        da_loss_func = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)
    elif args.da_method == 'dann':
        da_loss_func = AdversarialLoss(input_dim = 1024).to(device)
        # append discriminator classifier parameters to optimizer
        extra_params = da_loss_func.domain_classifier.parameters()
        optimizer.add_param_group({'params':extra_params})

    print (model)

    def train(epoch):
        model.train()

        total_loss = 0
        num_sample = 0

        rolling_avg_loss = []
        counter = 0

        t_train_loader_iter = iter(t_train_loader)

        for data_s in s_train_loader: # load source, target for DA training

            optimizer.zero_grad()
            
            data_s = data_s.to(device)
            out, graph_emb = model(data_s)

            if args.da_method != 'none':
                # Load target train dataset (without label)
                try:
                    data_t = next(t_train_loader_iter)
                except StopIteration:
                    t_train_loader_iter = iter(t_train_loader)
                    data_t = next(t_train_loader_iter)

                
                data_t = data_t.to(device)
                out_t, graph_emb_t = model(data_t)

                # Throw last incompleted batch
                if out.size(0) != out_t.size(0):
                    continue

                # out: 32x10
                # graph_emb: 32x1024, graph_emb_t: 32x1024
                if args.da_method == 'entropy':
                    da_loss = da_loss_func(out_t)

                else:
                    # DA loss based on feature distribution
                    da_loss = da_loss_func(graph_emb, graph_emb_t)

            out = log_softmax(out)
            nll_loss = nll_loss_func(out, data_s.y.long())

            counter += 1

            if args.da_method == 'none':
                combine_loss = nll_loss
            else:
                combine_loss = args.xe_loss_weight * nll_loss + args.da_loss_alpha * da_loss
                
            combine_loss.backward()
            total_loss += combine_loss.item() * data_s.num_graphs

            optimizer.step()

            # Do a test after certain amount of training samples
            num_sample = num_sample + len(data_s.y)


        return total_loss / len(train_dataset)


    def test(loader, disp_labels, epoch=-1):
        model.eval()

        correct = 0
        y_pred_all = []
        y_true_all = []
        for data in loader:
            
            data = data.to(device)
            with torch.no_grad():
                
                out, graph_emb = model(data)
                
                
                out = log_softmax(out)
                pred = out.max(dim=1)[1]

                
                y_pred = np.array(pred.cpu())
                y_true = np.array(data.y.cpu())


            correct += pred.eq(data.y).sum().item()
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)


        mcc_score_all  = matthews_corrcoef(y_true_all, y_pred_all)
        f1_score_all = f1_score(y_true_all, y_pred_all, average='weighted')
        
        acc = accuracy_score(y_true_all, y_pred_all)


        fig_save_path = os.path.join(result_dir, 'epoch'+str(epoch)+'_conf_matrix.png')
        save_confusion_matrix(y_true_all, y_pred_all, disp_labels=disp_labels, save_path=fig_save_path)

        # Save class-wise accuracy
        cmat = confusion_matrix(y_true_all, y_pred_all)
        class_acc = cmat.diagonal()/cmat.sum(axis=1)


        return acc, f1_score_all, mcc_score_all, class_acc.tolist()


    # Test the model before training
    disp_labels = t_test_loader.dataset.categories

    val_acc, val_f1_score_all, val_mcc_score_all, val_class_acc = test(t_val_loader, disp_labels)
    print('Epoch -1 (no training), Val Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}'.format(val_acc, val_f1_score_all, val_mcc_score_all))

    test_acc, f1_score_all, mcc_score_all, class_acc = test(t_test_loader, disp_labels)
    print('Epoch -1 (no training), Test Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}'.format(test_acc, f1_score_all, mcc_score_all))

    acc_valid = []
    best_acc = 0.0
    best_mcc = 0.0
    mcc_epoch = 0 # for loggin epoch of best mcc
    acc_epoch = 0 # for loggin epoch of best mcc
    num_epoch = args.epoch


    for epoch in range(1, num_epoch):

        loss = train(epoch)
        scheduler.step()

        # Eval each epoch
        print('---------------------------')
        val_acc, val_f1_score_all, val_mcc_score_all, val_class_acc = test(t_val_loader, disp_labels, epoch)
        print('Epoch {:03d}, Loss: {:.4f}, Val Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}'.format(epoch, loss, val_acc, val_f1_score_all, val_mcc_score_all))


        test_acc, test_f1_score_all, test_mcc_score_all, class_acc = test(t_test_loader, disp_labels, epoch)
        print('Epoch {:03d}, Loss: {:.4f}, Test Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}'.format(epoch, loss, test_acc, test_f1_score_all, test_mcc_score_all))
        

        acc_dict = {
              "epoch": epoch,
              "val_acc": val_acc,
              "val_f1": val_f1_score_all,
              "val_mcc": val_mcc_score_all,
              "val_class_acc": val_class_acc,
              "test_acc": test_acc,
              "test_f1": test_f1_score_all,
              "test_mcc": test_mcc_score_all,
              "test_class_acc": class_acc

        }

        acc_valid.append(acc_dict)
        acc_path = os.path.join(result_dir, 'acc_val.json')
        with open(acc_path, 'w') as fp:
            json.dump(acc_valid, fp)

        # Save Best
        if val_acc > best_acc:
        # save the model
            best_acc = val_acc
            acc_epoch = epoch
            checkpoint = os.path.join(result_dir, 'point_model_best_acc.pth')
            dict_save = {'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'args': args}
            torch.save(dict_save, checkpoint)
            print ('Best Val Acc Model Saved. Epoch', epoch)

            best_path = os.path.join(result_dir, 'best_acc_metrics.json')
            with open(best_path, 'w') as textfile:
                json.dump(acc_dict, textfile)
            textfile.close()

        
        if val_mcc_score_all > best_mcc:
            # save the model
            best_mcc = val_mcc_score_all
            mcc_epoch = epoch
            checkpoint = os.path.join(result_dir, 'point_model_best_mcc.pth')
            dict_save = {'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'args': args}
            torch.save(dict_save, checkpoint)
            print ('Best Val Mcc Model Saved. Epoch', epoch)

            best_path = os.path.join(result_dir, 'best_mcc_metrics.json')
            with open(best_path, 'w') as textfile:
                json.dump(acc_dict, textfile)
            textfile.close()

    best_results_logger = os.path.join(args.result_root, 'result_log.txt')
    file_obj = open(best_results_logger, 'a')
    file_obj.write('Name: %s, Epoch %d, Val Acc: %f\n' % (args.result_append, acc_epoch, best_acc))
    file_obj.write('Name: %s, Epoch %d, Val MCC: %f\n' % (args.result_append, mcc_epoch, best_mcc))
    file_obj.close()


if __name__ == "__main__":
    main()
