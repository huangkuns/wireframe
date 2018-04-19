import argparse
import os
import ref
import json

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    def init(self):
        self.parser.add_argument('--create_dataset', dest='create_dataset',
                                 action='store_true',
                                 help="create dataset")
        
        self.parser.add_argument('--json', dest='load_json',
                                 action='store_true',
                                 help="load arch parameters from file")
        
        self.parser.add_argument('--balance', dest='balance',
                                 action='store_true',
                                 help="balance the sample of positive and negative")
        
        self.parser.add_argument('--ratio', dest='ratio',
                                 default=7, type=float,
                                 help="ratio of positive and negative")
        
        self.parser.add_argument('--trainer', dest='trainer', type=str, 
                                 default='balance_junction',
                                 help="specify trainer name.")
        
        self.parser.add_argument('--criterion', dest='criterion', type=str, 
                                 default='balance',
                                 help="specify criterion name.")
        
        self.parser.add_argument('--decoder', dest='decoder', type=str, 
                                 default='junction',
                                 help="specify decoder name.")
        
        self.parser.add_argument('--split', dest='split', type=str, 
                                 default='train',
                                 help="specify dataset split.")

        #self.parser.add_argument('--hype', dest='hypefile', default='',
        #                         type=str, 
        #                         help='load hype-parameters from file')

        ## The exp name
        self.parser.add_argument('-e', '--exp', dest='exp', default='1', 
                                 help='experiment name')
        ## the model parameters
        self.parser.add_argument('--net', default='inception', dest='net', 
                                 help='specify feature network.')
        self.parser.add_argument('--test', action='store_true', dest='test',
                                 help='test or not')
        
        ## Input output
        self.parser.add_argument('--image_size', type=int, default=480, 
                                 dest='image_size',
                                 help='Input image dimension.')
        self.parser.add_argument('--grid_size', type=int, default=60, 
                                 dest='grid_size',
                                 help='output dimension.')
        self.parser.add_argument('--focus_size', type=float, default=1.5,
                                 dest='focus_size', 
                                 help='range one grid cell covers')
        self.parser.add_argument('--num_bin', type=int, default=15,
                                 dest='num_bin', 
                                 help='num of bins for 0~360')
        self.parser.add_argument('--max_len', type=int, default=1,
                                 dest='max_len', 
                                 help='max num of junction at each cell')
        self.parser.add_argument('--loss_weights', type=str, default='1.0, 0.1, 1.0, 0.1',
                                 dest='loss_weights', 
                                 help='loss weights for junction and bin losses')
        self.parser.add_argument('--decodeFeats', type=int, default=256,
                                 help='the channel number of decoder network')

        self.parser.add_argument('--valIntervals', type=int, default=5, 
                                 help='valid intervel')
        self.parser.add_argument('--gpu', type=str, default='0', 
                                 help='specify the gpu deivce(s) to use')
        self.parser.add_argument('--epochs', dest='max_epochs',
                                 help='number of epochs to train',
                                 default=17, type=int)
        self.parser.add_argument('--save_dir', dest='save_dir', 
                                 default="output",
                                 nargs=argparse.REMAINDER,
                                 help='directory to save models')
        self.parser.add_argument('--num_workers', dest='num_workers', 
                                 default=1, type=int, 
                                 help='number of worker to load data'
                                 )
        self.parser.add_argument('--batch_size','-b', dest='batch_size',
                                 help='batch_size',
                                 default=1, type=int)

        # config optimization
        self.parser.add_argument('-o', '--optimizer', dest='optimizer',
                                 help='training optimizer',
                                 default="sgd", type=str)
        self.parser.add_argument('--lr', dest='lr',
                                 help='starting learning rate',
                                 default=0.01, type=float)
        
        self.parser.add_argument('--lr_steps', dest='lr_steps',
                                 help='starting learning rate',
                                 default='8,12,16', type=str)
        
        self.parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                                 help='step to do learning rate decay',
                                 default=5, type=int)
        self.parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                                 help='learning rate decay ratio',
                                 default=0.1, type=float)
        self.parser.add_argument('--clip_norm', dest='clip_norm',
                                 help='clip norm when optimizing',
                                 default=1., type=float)

        # resume trained model
        self.parser.add_argument('-r', '--resume', dest='resume',
                                 help='resume checkpoint or not',
                                 default=False, type=bool)
        self.parser.add_argument('--checksession', dest='checksession',
                                 help='checksession to load model',
                                 default=1, type=int)
        self.parser.add_argument('--checkepoch', dest='checkepoch',
                                 help='checkepoch to load model',
                                 default=16, type=int)
        self.parser.add_argument('--checkdir', dest='checkdir',
                                 help='path of checkpoints are saved.',
                                 default='', type=str)
        self.parser.add_argument('--checkpoint', dest='checkpoint',
                                 help='checkpoint to load model',
                                 default=0, type=str)

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.saveDir = os.path.join(ref.output_root, self.opt.exp)
        self.opt.hypefile = os.path.join(ref.hypeDir, self.opt.exp + '.json')
        
        opt = self.opt
        if opt.load_json:
            with open(self.opt.hypefile, 'r') as f:
                H = json.load(f)
                self.opt.hype = H
            if opt.balance:
                H['balance'] = True
                H['ratio'] = opt.ratio
        else:
            H = {}
            H['image_size'] = opt.image_size
            H['grid_size'] = opt.grid_size
            region_size = int(opt.image_size // opt.grid_size)
            assert region_size * opt.grid_size == opt.image_size, "region_size x grid_size != image_size"
            H['region_size'] = region_size
            H['num_bin'] = opt.num_bin
            H['max_len'] = opt.max_len
            
            H['decodeFeats'] = opt.decodeFeats
            H['focus_size'] = opt.focus_size
            H['loss_weights'] = [float(x) for x in opt.loss_weights.split(',')]
            H['steppoints'] = [int(x for x in opt.lr_steps.split(','))]
            self.opt.hype = H

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        refs = dict((name, getattr(ref, name)) for name in dir(ref)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)
        
        return self.opt
