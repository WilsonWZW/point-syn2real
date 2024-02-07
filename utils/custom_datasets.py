import os
import os.path as osp
import shutil
import glob

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_off
from torch_geometric.data import Data
import torch_geometric.transforms as T

import h5py
import json
import numpy as np
import open3d as o3d
from plyfile import PlyData
from torch_geometric.data import Data

# copy from pytorch.geometric 1.3.2
def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)
    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa.astype(int), dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = Data(pos=pos, face=face)

    return data

def subset_spliter(pyg_dataset, val_rate = 0.1):
    # return train_split, val_split
    pass

class ModelNet(InMemoryDataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        assert name in ['10', '40']
        self.name = name
        super(ModelNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'ModelNet{}'.format(self.name))
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class ModelNet_DA_Subset(InMemoryDataset):
    def __init__(self, root, name='DA10_fullview', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']
        super(ModelNet_DA_Subset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        
        data_list = []
        for target, category in enumerate(self.categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.npy'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)

            for path in paths:

                point = np.load(path)
                label = target

                pos = torch.tensor(point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))


class ModelNet_Multiview(InMemoryDataset):
    def __init__(self, root, name='DA10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']
        super(ModelNet_Multiview, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        data_list = []
        for target, category in enumerate(self.categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.pcd'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            # Load pos, Label (y)

            for path in paths:

                pcd = o3d.io.read_point_cloud(path)
                point = np.asarray(pcd.points)
                label = target
                
                # -z is the zenith in modelnet_multiview/raw
                xx = point[:,0] 
                yy = point[:,1]
                zz = point[:,2]
                aligned_point = np.column_stack((xx, yy, -zz))

                pos = torch.tensor(aligned_point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)

                # Got some nauty near-empty data points
                if data.pos.shape[0] < 30:
                    print ('Invalid data, skipped', data.pos.shape)
                else:                
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class ShapeNet_DA_Subset(InMemoryDataset):
    def __init__(self, root, name='DA10_fullview', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']
        super(ShapeNet_DA_Subset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        
        data_list = []
        for target, category in enumerate(self.categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.npy'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            
            for path in paths:

                point = np.load(path)
                label = target

                # y is the zenith in shapenet_sub/raw
                xx = point[:,0] 
                yy = point[:,1]
                zz = point[:,2]
                if category == 'plant': # Plants are collected from ShapeNet semantic, diff orientation
                    aligned_point = np.column_stack((xx, yy, zz))
                else:
                    aligned_point = np.column_stack((xx, zz, yy))

                pos = torch.tensor(aligned_point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class ShapeNet_Multiview(InMemoryDataset):
    def __init__(self, root, name='5k', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']
        super(ShapeNet_Multiview, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        data_list = []
        for target, category in enumerate(self.categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.pcd'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            # Load pos, Label (y)

            for path in paths:

                pcd = o3d.io.read_point_cloud(path)
                point = np.asarray(pcd.points)
                label = target
                
                # -y is the zenith in shapenet_multiview/raw
                xx = point[:,0] 
                yy = point[:,1]
                zz = point[:,2]
                if category == 'plant': # Plants are collected from ShapeNet semantic, diff orientation
                    aligned_point = np.column_stack((xx, yy, -zz))
                else:
                    aligned_point = np.column_stack((xx, zz, -yy))

                pos = torch.tensor(aligned_point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)

                # Got some nauty near-empty data points
                if data.pos.shape[0] < 30:
                    print ('Invalid data, skipped', data.pos.shape)
                else:                
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class SimuGlaser_Multiview(InMemoryDataset):
    pass

# The simulated 3d city objects (maximum 4096-pts sampled point cloud)
class Simu3D_City(InMemoryDataset):

    def __init__(self, root, name='simu3d_city', train=True, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bicycle','car','motorcycle','person','truck']
        self.pre_transform = T.Compose([T.NormalizeScale(), T.SamplePoints(4096)])

        super(Simu3D_City, self).__init__(root, transform, self.pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bicycle','car','motorcycle','person','truck'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        data_list = []
        for target, category in enumerate(self.categories):

            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.ply'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            # Load pos, Label (y)

            for path in paths:
                data = read_ply(path) # example: Data(face=[3, 9662], pos=[13115, 3])

                data.y = torch.tensor([target])
                data = self.pre_transform(data)
                data_list.append(data)

        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

# The simulated 3d city objects (maximum 4096-pts to limit size. multiview sampled point cloud)
class Simu3D_City_Multiview(InMemoryDataset):
    def __init__(self, root, name='simu3d_city_multiview', train=True, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bicycle','car','motorcycle','person','truck']
        self.pre_transform = T.Compose([T.NormalizeScale(), T.FixedPoints(4096)])

        super(Simu3D_City_Multiview, self).__init__(root, transform, self.pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bicycle','car','motorcycle','person','truck'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        data_list = []
        for target, category in enumerate(self.categories):

            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.pcd'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            # Load pos, Label (y)

            for path in paths:


                pcd = o3d.io.read_point_cloud(path)
                point = np.asarray(pcd.points)
                label = target
                
                # z is the zenith
                xx = point[:,0] 
                yy = point[:,1]
                zz = point[:,2]
                aligned_point = np.column_stack((xx, yy, zz))

                pos = torch.tensor(aligned_point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)

                # Got some nauty near-empty data points
                if data.pos.shape[0] < 30:
                    print ('Invalid data, skipped', data.pos.shape)
                else:                
                    data_list.append(data)

        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class Scannet_ObjectDB(InMemoryDataset):
    def __init__(self, root, name='scannet10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']
        super(Scannet_ObjectDB, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
    
    # Scannet Healper function
    def load_dir(self, data_dir, name='train_files.txt'):
        with open(os.path.join(data_dir,name),'r') as f:
            lines = f.readlines()
        return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]

    @property
    def raw_file_names(self):
        return [
            'meta', 'test_0.h5', 'test_files.txt', 'train_0.h5', 'train_1.h5', 'train_2.h5', 'train_files.txt'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        
        if dataset == 'train':
            data_pth = self.load_dir(self.raw_dir, name='train_files.txt')
        else:
            data_pth = self.load_dir(self.raw_dir, name='test_files.txt')

        # Load all the points from h5py
        point_list = []
        label_list = []
        for pth in data_pth:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]
            
            point_list.append(point)
            label_list.append(label)
        scannet_data = np.concatenate(point_list, axis=0) # (6110, 2048, 6)
        scannet_label = np.concatenate(label_list, axis=0) # (6110, )

        if scannet_data.shape[0] == scannet_label.shape[0]:
            num_data = scannet_data.shape[0]
        else:
            raise ValueError('Number of data != label')

        data_list = []
        for idx, sn_data in enumerate(scannet_data):

            # y is the zenith in scannet/raw
            xx = sn_data[:, 0] 
            yy = sn_data[:, 1]
            zz = sn_data[:, 2]
            aligned_point = np.column_stack((xx, zz, yy))


            pos = torch.tensor(aligned_point, dtype=torch.float)
            y = torch.tensor([scannet_label[idx]])

            data = Data(pos=pos, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class SemKitti_Object(InMemoryDataset):
    def __init__(self, root, name='semkitti5', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.categories = ['bicycle','car','motorcycle','person','truck']
        super(SemKitti_Object, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bicycle','car','motorcycle','person','truck'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        data_list = []
        for target, category in enumerate(self.categories):

            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.pcd'.format(folder)) # when filename doesn't match folder name
            print('Processing %s folder...' % folder)
            # Load pos, Label (y)

            for path in paths:

                pcd = o3d.io.read_point_cloud(path)
                point = np.asarray(pcd.points)
                label = target
                
                # z is the zenith in semkitti raw
                xx = point[:,0] 
                yy = point[:,1]
                zz = point[:,2]
                
                aligned_point = np.column_stack((xx, yy, zz))

                pos = torch.tensor(aligned_point, dtype=torch.float)
                y = torch.tensor([label])
                data = Data(pos=pos, y=y)
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        col_data = self.collate(data_list)

        return col_data

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

class Glaser_Object(InMemoryDataset):
    pass
