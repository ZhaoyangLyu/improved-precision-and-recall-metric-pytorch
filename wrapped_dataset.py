import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np


class WrappedDataset(data.Dataset):
    def __init__(self, dataset, idx=None, return_tensor=True, transforms=None):
        self.dataset = dataset
        self.idx = idx # index of the tuple that we want from a sample from dataset
        self.return_tensor = return_tensor
        self.transforms = transforms
        self.len = len(dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.dataset.__getitem__(index)
        if self.idx is not None:
            x = x[self.idx]
        if self.return_tensor:
            x = torch.from_numpy(x)

        if not self.transforms is None:
            assert isinstance(self.transforms, list)
            for trans in self.transforms:
                x = trans(x)
        return x


class ImageFolderDataset(data.Dataset):
    def __init__(self, folder_path, transpose=True, normalize=False, return_image_name=False, rank=0, world_size=1):
        self.folder_path = folder_path
        self.transpose = transpose
        self.normalize = normalize
        self.return_image_name = return_image_name
        self.imgs = []
        valid_images = [".jpg",".png"]
        for f in os.listdir(self.folder_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_images:
                self.imgs.append(f)
            # imgs.append(Image.open(os.path.join(path,f)))

        self.len = len(self.imgs)
        print('Find %d images in the folder %s' % (self.len, self.folder_path))

        if world_size > 1:
            num_samples_per_rank = int(np.ceil(len(self.imgs) / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.imgs = self.imgs[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = len(self.imgs)

        self.len = len(self.imgs)
        print('This process hanldes %d images in the folder %s' % (self.len, self.folder_path))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.imgs[index]
        x = Image.open(os.path.join(self.folder_path, img_name))
        x = np.array(x)
        # unit8 type, range from 0 to 255, shape of (H,W,3)
        if self.transpose:
            x = x.transpose(2,0,1) # shape of (3,H,W)
        if self.normalize:
            x = x.astype(np.float32)
            x = x/255 *2 -1 # float type range from -1 to 1
        if self.return_image_name:
            return x, img_name
        else:
            return x


if __name__ == '__main__':
    import pdb
    path='celeba_64/img_align_celeba_64'
    dataset = ImageFolderDataset(path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    for i, data in enumerate(dataloader):
        print(data.shape)
        # pdb.set_trace()