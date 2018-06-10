import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import ref

def parse():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument('--genLine', dest='genLine', action='store_true', help='Generate line map.')

    parser.add_argument('--debug',     default=False, type=str2bool,   help='Debug mode')
    parser.add_argument('--manualSeed',default=0, type=int,        help='manual seed')
    parser.add_argument('--GPU',  default=True, type=str2bool,   help='Use GPU')

    parser.add_argument('--GPUs', default='0', type=str,        help='ID of GPUs to use, seperate by ,')

    parser.add_argument('--backend',          default='cudnn', type=str,        help='backend', choices=['cudnn', 'cunn'])
    parser.add_argument('--cudnn',            default='fastest',      type=str,
                        help='cudnn setting', choices=['fastest', 'deterministic', ' default'])
    # Path options
    parser.add_argument('--dataset', default='indoor', dest='dataset', type=str, help='dataset name')
    parser.add_argument('--resume',           default='../result/linepx/',
                        type=str,        help='Path to checkpoint and path to test result')
    # Data options
    parser.add_argument('--nThreads',         default=8,
                        type=int,        help='Number of data loading threads')
    parser.add_argument('--trainPctg',        default=0.95,
                        type=float,      help='Percentage of training images')
    parser.add_argument('--imgDim',           default=320,
                        type=int,        help='Image dimension')
    # Training/testing options
    parser.add_argument('--nEpochs',          default=200,
                        type=int,        help='Number of total epochs to run')
    parser.add_argument('--epochNum',         default=-1,             type=int,
                        help='0=retrain | -1=latest | -2=best', choices=[0, -1, -2])
    parser.add_argument('--batchSize',        default=4,
                        type=int,        help='mini-batch size')
    parser.add_argument('--saveEpoch',        default=10,
                        type=int,        help='saving at least # epochs')
    parser.add_argument('--testOnly',         default=False,
                        type=str2bool,   help='Run the test to see the performance')
    parser.add_argument('--barwidth',         default=50,
                        type=int,        help='Progress bar width')
    parser.add_argument('--visTrain',         default=15,
                        type=int,        help='Visualizing training examples')
    parser.add_argument('--visTest',          default=15,
                        type=int,        help='Visualizing testing examples')
    parser.add_argument('--visWidth',         default=1,
                        type=int,        help='Number of images per row for visualization')
    parser.add_argument('--visThres',         default=0.05,
                        type=float,      help='Threshold for visualization')
    # Optimization options
    parser.add_argument('--LR',               default=0.001,
                        type=float,      help='initial learning rate')
    parser.add_argument('--LRDecay',          default='exp',
                        type=str,        help='LRDecay method')
    parser.add_argument('--LRDParam',         default=100,
                        type=int,        help='param for learning rate decay')
    parser.add_argument('--momentum',         default=0.9,
                        type=float,      help='momentum')
    parser.add_argument('--dampening',        default=0.0,
                        type=float,      help='dampening')
    parser.add_argument('--weightDecay',      default=1e-4,
                        type=float,      help='weight decay')
    parser.add_argument('--optimizer',        default='SGD',          type=str,
                        help='optimizertype, more choices available', choices=['SGD', 'Adam'])
    # Model options
    parser.add_argument('--netType',          default='stackedHGB',
                        type=str,        help='Your defined model name')
    parser.add_argument('--netSpec',          default='custom',       type=str,
                        help='Other model to be loaded', choices=['custom', 'resnet'])
    parser.add_argument('--pretrain',         default=False,
                        type=str2bool,   help='Pretrained or not')
    parser.add_argument('--absLoss',          default=0,
                        type=float,      help='Weight for abs criterion')
    parser.add_argument('--bceLoss',          default=0,
                        type=float,      help='Weight for bce criterion')
    parser.add_argument('--mseLoss',          default=1,
                        type=float,      help='Weight for mse criterion')
    # Hourglass model options
    parser.add_argument('--nFeats',           default=256,
                        type=int,        help='Number of features in the hourglass')
    parser.add_argument('--nStack',           default=5,
                        type=int,        help='Number of hourglasses to stack')
    parser.add_argument('--nModules',         default=2,              type=int,
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--nOutChannels',     default=64,
                        type=int,        help='Number of output channels')
    # Other model options
    parser.add_argument('--numClasses',       default=4,
                        type=int,        help='Number of classes in the dataset')
    parser.add_argument('--suffix',           default='',
                        type=str,        help='Suffix for saving the model')

    opt = parser.parse_args()

    if opt.GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPUs
        cudnn.benchmark = True

    torch.set_default_tensor_type('torch.FloatTensor')

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.GPU:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if opt.debug:
        opt.nEpochs = 1
        opt.nThreads = 1
        opt.visTrain = min(opt.visTrain, 10)
        opt.visTest = min(opt.visTest, 10)

    opt.data = ref.data_folder

    return opt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
