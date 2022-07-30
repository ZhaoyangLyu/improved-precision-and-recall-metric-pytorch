import torch
import torch_fidelity

import argparse
import pickle

from wrapped_dataset import WrappedDataset, ImageFolderDataset
import os

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str, default='../diffusion_and_reverse/image_generation_exps/stylegan2_generated_celeba_128_imgs/model_trained_first_110_steps/ckpt_1000000_epoch_236/diffusion_and_reverse_images_from_t_100', help='the first folder, the folder that we want to evaluate metrics')

    parser.add_argument('--dataset', type=str, default='celeba128', help='the reference dataset, this will affect folder2 and folder2 cache name')
    # parser.add_argument('--folder2', type=str, default='../datasets/celeba_dataset/img_align_celeba_128', help='the second folder, this folder is usually used as the reference dataset')
    # parser.add_argument('--input2_cache_name', type=str, default='celeba128', help='the cache name of the second folder')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--device', type=str, default='0', help='index of the gpu to use')
    parser.add_argument('--save_file', type=str, default='evaluation_results/metric.pkl', help='the pickle file to save the computed metrics')
    # parser.add_argument("--clamp", action="store_true", help='whether clamp the noised image to -1 and 1')

    args = parser.parse_args()

    if args.dataset == 'celeba128':
        folder2 = 'datasets/celeba/img_align_celeba_128'
    elif args.dataset == 'celeba64':
        folder2 = 'datasets/celeba/img_align_celeba_64'
    elif args.dataset == 'cifar10':
        folder2 = 'datasets/cifar10'
    else:
        raise Exception('Dataset %s is not supported' % args.dataset)

    dataset1 = ImageFolderDataset(args.folder1, transpose=True)
    dataset2 = ImageFolderDataset(folder2, transpose=True)

    dataset1 = WrappedDataset(dataset1, idx=None, return_tensor=True, transforms=None)
    dataset2 = WrappedDataset(dataset2, idx=None, return_tensor=True, transforms=None)
    
    # compute fid
    if args.device == 'none':
        print('User has not specified a cuda device')
        print('Using the system (Slurm) allocated device', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    input2_cache_name = args.dataset#args.input2_cache_name if len(args.input2_cache_name) > 0 else None
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dataset1, 
        input2=dataset2, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=True, 
        verbose=False,
        batch_size = args.batch_size,
        input2_cache_name=input2_cache_name
    )
    print(metrics_dict)
    
    if len(args.save_file) > 0:
        save_dir = os.path.split(args.save_file)[0]
        os.makedirs(save_dir, exist_ok=True)
        print('saving computed metrics to file', args.save_file)
        with open(args.save_file, 'wb') as handle:
            pickle.dump(metrics_dict, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    # pdb.set_trace()