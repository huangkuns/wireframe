# Code for training junction detector.
## Downloading Data


## Preprocess
- install requirements in python3.
- put hype-parameter config file in hypes/EXPNAME.json, follow the example.
- preprocess dataset
```
python main.py --exp 1 --json --create_dataset
```

## Training
- train junction detector.
```
  python3 main.py --exp 1 --json --gpu 0
```
- testing.
```
   python3 main.py --exp 1 --json --test --checkepoch 16 --gpu 0
```
