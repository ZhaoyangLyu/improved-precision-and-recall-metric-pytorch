import torch
import torchvision
import os
import pdb
from tqdm import tqdm

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10('./', train=True, transform=None, target_transform = None, download = True)
    save_path = './datasets/cifar10'
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(len(dataset))):
        x, label = dataset.__getitem__(i)
        save_name = '%d_label_%d.png' % (i, label)
        save_name = os.path.join(save_path, save_name)
        x.save(save_name)
        # pdb.set_trace()