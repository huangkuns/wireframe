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
                        str, default='', help='labels list to plot')
    args = parser.parse_args()
    exps = args.exps
    labels = args.labels
    exps = exps.split(',')
    labels = labels.split(',') if labels else ['' for _ in exps]
    print(exps, labels)

    plots = []
    plot_labels = []
    
    for exp, label in zip(exps, labels):
        mat = sio.loadmat('{}.mat'.format(exp))
        prs = mat['sumprecisions']
        rcs = mat['sumrecalls']
        N = mat['nsamples']
        prs = prs[:, 0]
        rcs = rcs[:, 0]
        N = N[:, 0]
        prs = [x / float(n) for x, n in zip(prs, N)]
        rcs = [x / float(n) for x, n in zip(rcs, N)]
        line_x = plt.plot(rcs, prs, '-*', linewidth=2, label=label if label else exp)
    plt.grid(True)
    plt.axis([0., 1., 0., 1.])
    plt.xticks(np.arange(0, 1., step=0.1))
    plt.yticks(np.arange(0, 1., step=0.1))
    plt.legend(loc=0)
    plt.savefig('{}.png'.format('_'.join(exps)), dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
