import sys
sys.path.append("..")
import datasets.init_dataset as init_dataset
from torch.utils.data.dataloader import *


class myDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
        DataLoader.__init__(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last)


def create(opt):
    loaders = []
    for split in ['train', 'val']:
        dataset = init_dataset.create(opt, split)
        if split == 'train':
            loaders.append(myDataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads))
        elif split == 'val':
            loaders.append(myDataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads))
    return loaders[0], loaders[1]
