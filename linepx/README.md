## Train [straight] line pixel detector here.

-----

Folders
----
| Folder     | Description                  |
|------------|------------------------------|
| models      | network architectures and parameters |
| datasets   | define dataloader interfaces |
| criterions | define criterion interfaces  |
| util       | visulization utilities       |

Preprocessing
-----
Please run `python main.py -h` for the general usage of the code.

To preprocess the images and annotations:

`python3 main.py --genLine  # training data`


`python3 main.py --genLine --testOnly t  # test data`


Training
-----


To train the network for our dataset on 1 GPU:

`python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 4`

Testing
-----

To test the best model:

`python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --testOnly t`
