import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import collections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exps', '-e', dest='exps', type=
                        str, default='', help='exps list to plot')
    parser.add_argument('--labels', '-l', dest='labels', type=
                        str, default='', help='labels assigned to exps list in legend')

    args = parser.parse_args()
    exps = args.exps
    labels = args.labels
    exp_iters = [x for x in exps.split(',')]
    if labels:
        labels = [x for x in labels.split(',')]
    else:
        labels = ['' for _ in exp_iters]
    # print('_'.join(labels))
    
    mats = {x:(sio.loadmat('plots/{}.mat'.format(x)), label) 
            for x, label in zip(exp_iters, labels)}
    plots = []
    plot_labels = []
    orderMats = collections.OrderedDict(sorted(mats.items()))
    for x, lv in orderMats.items():
        v, label = lv
        # print(v.keys())
        prs = v['pr_iter'][0, :]
        rcs = v['rc_iter'][0, :]
        line_x = plt.plot(rcs, prs, '-*', linewidth=2, label=label if label else x)
        plots += [line_x]
    plt.grid(True)
    plt.axis([0., 1., 0., 1.])
    plt.xticks(np.arange(0, 1., step=0.1))
    plt.yticks(np.arange(0, 1., step=0.1))
    #plt.legend(bbox_to_anchor=(0.9, 0.78), bbox_transform=plt.gcf().transFigure)
    plt.legend(loc=0)
    #plt.savefig('junc_{}.pdf'.format('_'.join(orderMats.keys())), dpi=200)
    plt.savefig('junc_{}.png'.format('_'.join(orderMats.keys())), dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
