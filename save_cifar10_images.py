import torch
import torchvision
import pdb

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10('./', train=True, transform=None, target_transform = None, download = True)

    for i in range(len(dataset)):
        x = dataset.__getitem__(i)
        pdb.set_trace()