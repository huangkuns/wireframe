import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def draw(dirPath, plotPath):
    loss = []
    tloss = []
    trainLog = os.path.join(dirPath, 'train.log')
    testLog = os.path.join(dirPath, 'test.log')

    if not os.path.exists(trainLog):
        print('Error opening train.log')
    if not os.path.exists(testLog):
        print('Error opening test.log')

    with open(trainLog, 'r') as f:
        log = f.read()
        for line in log.split('\n'):
            if 'Finished' in line:
                loss.append(float(line.split(':')[1].split()[0]))

    with open(testLog, 'r') as f:
        log = f.read()
        for line in log.split('\n'):
            if 'Finished' in line:
                tloss.append(float(line.split(':')[1].split()[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss, label='Train')
    ax.plot(tloss, label='Test')
    # plt.axis([0, 50, 0, 0.015])
    ax.legend()
    fig.savefig(os.path.join(plotPath, dirPath.split('/')[-1] + '.png'))
    plt.close()

    return min(tloss)

modelPath = '../models'
# for d in os.listdir(modelPath):
#     plotPath = os.path.join(modelPath, 'loss')
#     if not os.path.exists(plotPath):
#         os.makedirs(plotPath)
#
#     dirPath = os.path.join(modelPath, d)
#     if os.path.isdir(dirPath) and 'loss' not in d:
#         try:
            # minLoss = draw(dirPath, plotPath)
minLoss = draw(modelPath, modelPath)
            # print(d + ' finished with minimum loss: ' + str(minLoss))
print(' finished with minimum loss: ' + str(minLoss))
        # except:
        #     print('Cannot parse ' + dirPath)
