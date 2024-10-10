#!/usr/bin /python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_graph_utils import train_utils
import time

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='SGWN', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=1024, help='batchsize of the training process')
    parser.add_argument('--per_node', type=int, default=10, help='the number of nodes of each subgraph')
    parser.add_argument('--SNR', type=int, default=0, help='Signal to Noise Ratio')
    parser.add_argument('--data_name', type=str, default='XJTUSpurgearRadius', help='the name of the data')
    parser.add_argument('--Input_type', choices=['TD', 'FD','other'],type=str, default='TD', help='the input type decides the length of input')
    parser.add_argument('--data_dir', type=str, default= "E:\data\XJTU_Spurgear\XJTUSpurgearRadius.pkl", help='the directory of the data')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=100, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Define the tasks
    parser.add_argument('--task', choices=['Node', 'Graph'], type=str,
                        default='Graph', help='Node classification or Graph classification')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')
    # hyperparameter of SGWN
    parser.add_argument('--nhid', type=int, default=1024,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--s', type=int, default=2,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    if args.task == 'Node':
        sub_dir = args.task + '_' +args.model_name+ '_' + str(args.SNR) + '_'+args.data_name + '_' + args.Input_type +'_'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    else:
        sub_dir = args.task + '_' +args.model_name + '_' + str(args.SNR) + '_' + args.data_name + '_' + args.Input_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()



