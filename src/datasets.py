import os, torch
import torchvision
import numpy as np
import scipy.io as sio

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from smallNORB_transforms import *

def smallnorb(args, dataset_paths):

    transf = {'train': Compose([
                ToPILImage(),
                RandomCrop((args.crop_dim, args.crop_dim)),
                ColorJitter(brightness=args.brightness/255., contrast=args.contrast),
                ToTensor(),
                smallnorbStandardize()]),
                # Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))]),
        'test':  Compose([
                ToPILImage(),
                CenterCrop((args.crop_dim, args.crop_dim)),
                ToTensor(),
                smallnorbStandardize()])}
                # Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])}

    config = {'train': True, 'test': False}
    datasets = {i: DatasetLoader(dataset_paths[i], transform=transf[i],
        shuffle=config[i]) for i in config.keys()}

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].labels,
        n_classes=args.arch[-1],
        n_samples_per_class=np.unique(
            datasets['train'].labels, return_counts=True)[1] // 5) # % of train set per class

    # define transforms for train set (without valid data)
    transf['train_'] = Compose([
            ToPILImage(),
            RandomCrop((args.crop_dim, args.crop_dim)),
            ColorJitter(brightness=args.brightness/255., contrast=args.contrast),
            ToTensor(),
            smallnorbStandardize()])
            # Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # define transforms for class-balanced valid set
    transf['valid'] = Compose([
            ToPILImage(),
            CenterCrop((args.crop_dim, args.crop_dim)),
            ToTensor(),
            smallnorbStandardize()])
            # Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = smallNORBDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = smallNORBDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], pin_memory=True,
        num_workers=8, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

class DatasetLoader(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, data_path, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.data, self.labels = [], []

        # get path for each class folder
        for class_label_idx, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)

            # get name of each file per class and respective class name/label index
            for _, file_name in enumerate(os.listdir(class_path)):
                self.data.extend([np.load(os.path.join(data_path, class_name, file_name))])
                self.labels.append(class_label_idx)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        if self.shuffle:
            # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        image = self.data[idx]

        # transform each instance independently per channel
        if self.transform:
            image0, image1 = self.transform(image[0], image[1])

        return torch.cat((image0, image1)), self.labels[idx] # (X, Y)

class smallNORBDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.transform = transform

        # shuffle the dataset
        idx = np.random.permutation(self.data.shape[0])
        self.data = self.data[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img0, img1 = self.transform(self.data[idx][0], self.data[idx][1])
        return torch.cat((img0, img1)), self.labels[idx]

def mnist(args, dataset_paths):

    transf = {'train': transforms.Compose([
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ]),
        'test':  transforms.Compose([
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])}

    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.MNIST(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}

    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=args.arch[-1],
        n_samples_per_class=np.repeat(500, args.arch[-1])) # 500 per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], num_workers=8, pin_memory=True,
        batch_size=args.batch_size, shuffle=config[i]) for i in config.keys()}

    if args.test_affNIST:
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', 'affNIST')

        aff_transf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))])

        datasets['affNIST_test'] = affNIST(data_path=os.path.join(working_dir,'test'),
            transform=aff_transf)
        dataloaders['affNIST_test'] = DataLoader(datasets['affNIST_test'], pin_memory=True,
            num_workers=8, batch_size=args.batch_size, shuffle=False)

    return dataloaders

class affNIST(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            image, label: sample data and respective label'''

    def __init__(self, data_path, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.split = self.data_path.split('/')[-1]

        if self.split == 'train':
            for i, file in enumerate(os.listdir(data_path)):
                # load dataset .mat file batch
                self.dataset = sio.loadmat(os.path.join(data_path, file))
                # concatenate the 32 .mat files to make full dataset
                if i == 0:
                    self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
                    self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0])
                else:
                    self.data = np.concatenate((self.data,
                        np.array(self.dataset['affNISTdata']['image'][0][0])), axis=1)
                    self.labels = np.concatenate((self.labels,
                        np.array(self.dataset['affNISTdata']['label_int'][0][0])), axis=1)

            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            self.labels = self.labels.squeeze()
        else:
            # load valid/test dataset .mat file
            self.dataset = sio.loadmat(os.path.join(self.data_path, self.split+'.mat'))
            # (40*40, N)
            self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0]).squeeze()

        self.data = self.data.squeeze()

        if self.shuffle: # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        image = self.data[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[idx] # (X, Y)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]

def random_split(data, labels, n_classes, n_samples_per_class):
    ''' create a class-balanced validation set '''
    train_X, train_Y, valid_X, valid_Y = [],[],[],[]

    for c in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == c).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[c], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_X.extend(data[train_samples])
        train_Y.extend(labels[train_samples])
        valid_X.extend(data[valid_samples])
        valid_Y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # .stack transforms list of tensors to tensor
        return {'train': torch.stack(train_X), 'valid': torch.stack(valid_X)}, \
            {'train': torch.stack(train_Y), 'valid': torch.stack(valid_Y)}
    else:
        return {'train': train_X, 'valid': valid_X}, \
            {'train': train_Y, 'valid': valid_Y}
