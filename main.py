import argparse, os, torch
from functions import precision_and_recall, realism
from typing import Union

G_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'truncation_0_7')
R_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images1024x1024')

def parse_args():
    desc = "calcualte precision and recall OR realism"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--cal_type', type=str, default='precision_and_recall', choices=['precision_and_recall', 'realism'], help='The type of calcualtion')
    
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device', type=str, default='0', help='index of the gpu to use')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=str, default='50000', help='number of sample to consider, could also be set to string all t0 handle all samples')

    parser.add_argument('--generated_dir', default=G_DIRECTORY)
    parser.add_argument('--real_dir', default=R_DIRECTORY)
    parser.add_argument('--cache', action='store_true', help='whether to cache features and use cached features for real images from cifar10, celeba64, celeba128')

    parser.add_argument('--dataset', type = str, default='none', help='could be cifar10, celeba64, celeba128, or none')
    print(parser.parse_args())
    return check_args(parser.parse_args())


def check_args(args):
    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)    
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():
    # parse arguments
    args = parse_args()

    datasets_path = {'cifar10':'datasets/cifar10', 'celeba64':'datasets/celeba/img_align_celeba_64', 'celeba128':'datasets/celeba/img_align_celeba_128'}
    print('dataset is', args.dataset)
    if args.dataset in ['cifar10', 'celeba64', 'celeba128']:
        args.real_dir = datasets_path[args.dataset]
        print('using ref images from the directory', args.real_dir)

    if args.device == 'none':
        print('User has not specified a cuda device')
        print('Using the system (Slurm) allocated device', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.data_size == 'all':
        args.data_size = int(args.data_size)

    if args.cal_type == 'precision_and_recall':
        task = precision_and_recall(args)
    else:
        task = realism(args)

    task.run()

    print("Job finised!")

if __name__ == '__main__':
    '''
    python main.py --generated_dir /home/xuxudong/zylyu_2196/ddpms/My_DDIM/diffusion_and_reverse/image_generation_exps/stylegan2_generated_celeba_64_imgs/model_trained_first_210_steps/ckpt_45000_epoch_28/diffusion_and_reverse_images_from_t_200_var_type_fixedlarge --real_dir /home/xuxudong/zylyu_2196/ddpms/My_DDIM/diffusion_and_reverse/image_generation_exps/stylegan2_generated_celeba_64_imgs/model_trained_first_210_steps/ckpt_15000_epoch_9/diffusion_and_reverse_images_from_t_200_var_type_fixedlarge --data_size 100

    python main.py --generated_dir /home/xuxudong/zylyu_2196/ddpms/My_DDIM/diffusion_and_reverse/image_generation_exps/stylegan2_generated_celeba_64_imgs/model_trained_first_210_steps/ckpt_45000_epoch_28/diffusion_and_reverse_images_from_t_200_var_type_fixedlarge --real_dir /home/xuxudong/zylyu_2196/ddpms/My_DDIM/diffusion_and_reverse/image_generation_exps/stylegan2_generated_celeba_64_imgs/model_trained_first_210_steps/ckpt_45000_epoch_28/diffusion_and_reverse_images_from_t_200_var_type_fixedlarge --data_size 100

    python main.py --generated_dir ./datasets/cifar10 --real_dir ./datasets/cifar10 --data_size 100

    python main.py --generated_dir /home/xuxudong/zylyu_2196/ddpms/stylegan2-ada-pytorch/cifar10_stylegan_ada_unconditional/images  --data_size 50000 --dataset cifar10 --cache

    python main.py --generated_dir /home/zylyu/new_pool/ddpms/My_DDIM/diffusion_and_reverse/image_generation_exps/celeba_64_large_model_trained_full_1000_steps/ckpt_300000_epoch_189  --data_size all --dataset celeba64 --cache

    python main.py --generated_dir /home/zylyu/new_pool/ddpms/pytorch_diffusion_cifar10/pytorch_diffusion/results/ema_cifar10  --data_size all --dataset cifar10 --cache
    '''
    main()   