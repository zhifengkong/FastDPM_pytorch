import os
import torch
import argparse

from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument('-name', '--name', type=str, choices=["cifar10", "lsun_bedroom", "celeba64"],
                        help='Name of experiment')
    parser.add_argument('-ema', '--ema', action='store_true', help='Whether use ema')

    # fast generation parameters
    parser.add_argument('-approxdiff', '--approxdiff', type=str, choices=['STD', 'STEP', 'VAR'], help='approximate diffusion process')
    parser.add_argument('-kappa', '--kappa', type=float, default=1.0, help='factor to be multiplied to sigma')
    parser.add_argument('-S', '--S', type=int, default=50, help='number of steps')
    parser.add_argument('-schedule', '--schedule', type=str, choices=['linear', 'quadratic'], help='noise level schedules')

    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='gpu device')

    args = parser.parse_args()
    
    kwargs = {'batch_size': 50, 'device': torch.device('cuda:{}'.format(args.gpu)), 'dims': 2048}

    if args.approxdiff == 'STD':
        variance_schedule = '1000'
    else:
        variance_schedule = '{}{}'.format(args.S, args.schedule)
    folder = '{}{}_{}{}_kappa{}'.format('ema_' if args.ema else '',
                                        args.name, 
                                        args.approxdiff,
                                        variance_schedule,
                                        args.kappa)
    if folder not in os.listdir('generated'):
        raise Exception('folder not found')

    paths = ['./generated/{}'.format(folder),
            './pytorch_fid/{}_train_stat.npy'.format(args.name)]
    fid = calculate_fid_given_paths(paths=paths, **kwargs)
    print('{}: FID = {}'.format(folder, fid))