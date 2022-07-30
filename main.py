import argparse, os, torch
from functions import precision_and_recall, realism

G_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'truncation_0_7')
R_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images1024x1024')

def parse_args():
    desc = "calcualte precision and recall OR realism"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--cal_type', type=str, default='precision_and_recall', choices=['precision_and_recall', 'realism'], help='The type of calcualtion')
    
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=50000)

    parser.add_argument('--generated_dir', default=G_DIRECTORY)
    parser.add_argument('--real_dir', default=R_DIRECTORY)

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
    '''
    main()   