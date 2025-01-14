# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author seungwook, adopted from baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import getpass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_all_tf_combs, dataset_num_classes, \
    knn_monitor

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, true_labels, aug_labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            true_labels = true_labels.cuda()
            aug_labels = aug_labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs, outputs_online = net(images)
        loss = loss_function(outputs, aug_labels)
        loss_online = loss_function(outputs_online, true_labels)
        loss_total = loss + loss_online
        loss_total.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)


        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLoss Online Clf: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                loss_online.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True, num_aug_classes=0):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    test_loss_online = 0.0
    correct = 0.0
    correct_online = 0.0
    correct_per_class = torch.zeros(num_aug_classes, device='cuda')
    total_per_class = torch.zeros(num_aug_classes, device='cuda')

    for (images, true_labels, aug_labels) in cifar100_test_loader:

        if args.gpu:
            true_labels = true_labels.cuda()
            aug_labels = aug_labels.cuda()
            images = images.cuda()

        outputs, outputs_online = net(images)
        loss = loss_function(outputs, aug_labels)
        loss_online = loss_function(outputs_online, true_labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(aug_labels).sum()

        test_loss_online += loss_online.item()
        _, preds_online = outputs_online.max(1)
        correct_online += preds_online.eq(true_labels).sum()

        # mean per class accuracy
        correct_vec = (preds == aug_labels) # if each prediction is correct or not
        ind_per_class = (aug_labels.unsqueeze(1) == torch.arange(num_aug_classes, device='cuda')) # indicator variable for each class
        correct_per_class += (correct_vec.unsqueeze(1) * ind_per_class).sum(0)
        total_per_class += ind_per_class.sum(0)

    # sanity check that the sum of total per class amounts to the whole dataset
    assert total_per_class.sum() == len(cifar100_test_loader.dataset)
    acc_per_class = correct_per_class / total_per_class

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Mean per-class accuracy: {:.4f}, Average online clf loss: {:.4f}, Online clf accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        acc_per_class.mean().float(),
        test_loss_online / len(cifar100_test_loader.dataset),
        correct_online.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Mean per class accuracy', acc_per_class.mean().float(), epoch)
        writer.add_scalar('Test/Average online clf loss', test_loss_online / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy online clf', correct_online.float() / len(cifar100_test_loader.dataset), epoch)

        for c in range(num_aug_classes):
            writer.add_scalar(f'Test/Class {c} ({str(all_tf_combs[c])}) accuracy', acc_per_class[c].float(), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

def make_sh_and_submit(args, delay=0):
    os.makedirs('./scripts/submit_scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)

    # cleanup
    sys.argv.remove('--submit')
    options = ' '.join(sys.argv[1:])

    # setting experiment name from some params
    args.exp_name = ''
    for a in ['net', 'dataset', 'batch_size', 'lr', 'tfs', 'max_num_tf_combos']:
        if a == 'tfs':
            v = '_'.join(getattr(args, a))
        else:
            v = getattr(args, a)
        args.exp_name += f'{a}_{v}_'

    print(f'Submitting the job with options: {args}')

    # supercloud slurm config
    # username = getpass.getuser()
    preamble = (
        f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
        f'-o ./logs/{args.exp_name}.out\n#SBATCH '
        f'--job-name={args.exp_name}\n#SBATCH '
        f'--open-mode=append\n\n'
    )

    with open(f'./scripts/submit_scripts/{args.exp_name}_{delay}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        file.write(
            f'python {sys.argv[0]} '
            f'{options}'
        )

    os.system(f'sbatch ./scripts/submit_scripts/{args.exp_name}_{delay}.sh')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--data', type=str, default='/data/scratch/swhan/data/', help='path to data directory')
    parser.add_argument('--dataset', type=str, default='cifar100', help='name of dataset')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--online-clf', action='store_true', default=False, help='monitor online classifier test accuracy')
    parser.add_argument('--tfs',  nargs='+', default=[], help='Choose from [crop, hflip, vflip, rotate, invert, blur, solarize, grayscale, colorjitter, halfswap')
    parser.add_argument('--max-num-tf-combos', type=int, default=-1, help='Maximum number of augmentation combination per class (-1 is all)')

    # kNN args
    parser.add_argument('--knn-monitor', action='store_true', default=False, help='monitor knn test accuracy')
    parser.add_argument('--knn-int', type=int, default=1, help='interval (in # of epochs) to perform kNN monitor')
    
    # supercloud args
    parser.add_argument('--submit', action='store_true', default=False, help='whether to submit it as a slurm job')

    args = parser.parse_args()

    if args.submit:
        make_sh_and_submit(args)
        sys.exit(0)

    all_tf_combs = get_all_tf_combs(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, args.tfs, args.max_num_tf_combos)
    test_tf = get_all_tf_combs(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD, [], 0)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        args.data,
        all_tf_combs,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    # train loader used as memory bank for knn monitor (only default transformations)
    cifar100_memory_loader = get_training_dataloader(
        args.data,
        test_tf,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False
    )

    cifar100_test_loader = get_test_dataloader(
        args.data,
        all_tf_combs,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # test loader used as memory bank for knn monitor (only default transformations)
    cifar100_default_test_loader = get_test_dataloader(
        args.data,
        test_tf,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f'Initializing {args.net} with {len(all_tf_combs)} number of augmented classes')
    net = get_network(args, num_classes=len(all_tf_combs), online_num_classes=dataset_num_classes[args.dataset])

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)
    writer.add_text('Transformations', str(all_tf_combs))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch, num_aug_classes=len(all_tf_combs))

        if (epoch % args.knn_int) == 1:
            knn_acc = knn_monitor(net, cifar100_memory_loader, cifar100_default_test_loader, 'cuda', k=200, writer=writer, epoch=epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
