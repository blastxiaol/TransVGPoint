import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate
import yaml

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('config', type=str)
    parser.add_argument('eval_model', type=str)
    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--debug', action='store_true')


    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_test = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)

    collate_fn = utils.collate_fn3d if args.type == '3d' else utils.collate_fn2d
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    
    start_time = time.time()
    
    # perform evaluation
    accuracy50, accuracy25, miou = evaluate(args, model, data_loader_test, device)
    
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        log_stats = {'test_model:': args.eval_model,
                    'accu50:': accuracy50,
                    'accu25:': accuracy25,
                    'miou:': miou,
                    }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
                with (output_dir / "eval_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    if args.debug:
        args.output_dir = 'debug'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
