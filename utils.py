""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
import random
import time

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from itertools import combinations
from PIL import ImageOps

from dataset import AugmentedDataset

feature_dims = {
    'renset18': 512,
    'resnet50': 2048
}

dataset_num_classes = {
    'cifar100': 100,
}


def get_network(args, num_classes=100, online_num_classes=100):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes, online_num_classes=online_num_classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=num_classes, online_num_classes=online_num_classes)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes, online_num_classes=online_num_classes)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes, online_num_classes=online_num_classes)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=num_classes, online_num_classes=online_num_classes)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_all_tf_combs(mean, std, tfs, max_num_comb=-1):
    """ return all possible tf combinations
    Args:
        mean: mean of training dataset
        std: std of training dataset
    Returns: all possible combinations of transformations that defines each class
    """

    flexible_tf = []

    if 'crop' in tfs:
        flexible_tf.append(transforms.RandomCrop(32, padding=4))
    if 'hflip' in tfs:
        flexible_tf.append(transforms.RandomHorizontalFlip(p=1.0))
    if 'vflip' in tfs:
        flexible_tf.append(transforms.RandomVerticalFlip(p=1.0))
    if 'rotate' in tfs:
        flexible_tf.append(transforms.RandomRotation(90))
    if 'invert' in tfs:
        flexible_tf.append(transforms.RandomInvert(p=1.0))
    if 'blur' in tfs:
        flexible_tf.append(transforms.GaussianBlur(3, sigma=(0.1, 2.0)))
    if 'solarize' in tfs:
        flexible_tf.append(Solarization(1.0))
    if 'grayscale' in tfs:
        flexible_tf.append(transforms.Grayscale(3))
    if 'colorjitter' in tfs:
        flexible_tf.append(transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1))

    # TODO: halfswap (but this needs to work in the tensor space not PIL Image)

    # flexible_tf = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(p=1.0),
    #     transforms.RandomVerticalFlip(p=1.0),
    #     transforms.RandomRotation(90),
    #     transforms.RandomInvert(p=1.0),
    #     transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    #     ]
    default_tf = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]
    
    # getting all possible 1-n combinations of transformations
    all_tf_combs = []
    if max_num_comb == -1:
        max_num_comb = len(flexible_tf)
    
    for i in range(max_num_comb+1):
        combs = list(combinations(flexible_tf, i))
        all_tf_combs += combs
    
    # adding in default transformations
    for j in range(len(all_tf_combs)):
        all_tf_combs[j] = transforms.Compose(all_tf_combs[j] + tuple(default_tf))

    return all_tf_combs

    
def get_training_dataloader(data_dir, all_tfs, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        data_dir: path to data directory
        all_tfs: list of transformation combinations
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    # transform_train = transforms.Compose([
    #     #transforms.ToPILImage(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])

    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    # cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training = AugmentedDataset(data_dir, transform_list=all_tfs, train=True)
    
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(data_dir, all_tfs, batch_size=16, num_workers=2, shuffle=True):
    """ return testing dataloader
    Args:
        data_dir: path to data directory
        all_tfs: list of transformation combinations
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_data_loader:torch dataloader object
    """

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    # #cifar100_test = CIFAR100Test(path, transform=transform_test)
    # cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    cifar100_test = AugmentedDataset(data_dir, transform_list=all_tfs, train=False)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

##################
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None, epoch=0, writer=None):
    """
        kNN monitor
    """
    start = time.time()
    if not targets:
        targets = memory_data_loader.dataset.dataset.targets
    net.eval()
    classes = len(memory_data_loader.dataset.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    
    with torch.no_grad():
        # generate feature bank
        for data, target, _ in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True), extract_features=True)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target, _ in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data, extract_features=True)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0].cuda() == target).float().sum().item()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test kNN: Epoch: {}, kNN Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        total_top1 / total_num,
        finish - start
    ))
    print()

    if writer:
        writer.add_scalar('Test/kNN Accuracy', total_top1 / total_num, epoch)

    return total_top1 / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, k, t):

    feature_labels_pred = torch.empty((feature.shape[0], 1))
    dists = compute_distances_no_loops(feature_bank, feature)

    # Index over test images
    for i in range(dists.shape[1]):
        # Find index of k lowest values
        x = torch.topk(dists[:,i], k, largest=False).indices

        # Index the labels according to x
        k_lowest_labels = feature_labels[x]

        # y_test_pred[i] = the most frequent occuring index
        feature_labels_pred[i] = torch.argmax(torch.bincount(k_lowest_labels))
    
    return feature_labels_pred

def compute_distances_no_loops(x_train, x_test):
    """
    Inputs:
    x_train: shape (num_train, C, H, W) tensor.
    x_test: shape (num_test, C, H, W) tensor.

    Returns:
    dists: shape (num_train, num_test) tensor where dists[i, j] is the
        Euclidean distance between the ith training image and the jth test
        image.
    """
    # Get number of training and testing images
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    # Create return tensor with desired dimensions
    dists = x_train.new_zeros(num_train, num_test) # (500, 250)

    # Flattening tensors
    train = x_train.flatten(1) # (500, 3072)
    test = x_test.flatten(1) # (250, 3072)

    # Find Euclidean distance
    # Squaring elements
    train_sq = torch.square(train)
    test_sq = torch.square(test)

    # Summing row elements
    train_sum_sq = torch.sum(train_sq, 1) # (500)
    test_sum_sq = torch.sum(test_sq, 1) # (250)

    # Matrix multiplying train tensor with the transpose of test tensor
    mul = torch.matmul(train, test.permute(1, 0)) # (500, 250)

    # Reshape enables proper broadcasting.
    # train_sum_sq = [500, 1] shape tensor and test_sum_sq = [1, 250] shape tensor.
    # This enables broadcasting to match desired dimensions of dists
    dists = torch.sqrt(train_sum_sq.reshape(-1, 1) + test_sum_sq.reshape(1, -1) - 2*mul)

    return dists