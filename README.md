# CutMix

## Requirements

- Python 3
- Torch >= 1.1.0

## Setup

```bash
$ git install https://github.com/dsc126/cutmix.git
$ pip install -r requirements.txt
$ python setup.py install
```

## Training

```bash
$ python train.py -c conf/cifar100_pyramid200.yaml --expname [expname] --mode [mode] --tensorboard
```

eg. 

```bash
$ python train.py -c conf/cifar100_pyramid200.yaml --expname CutMix --mode CutMix --tensorboard
$ python train.py -c conf/cifar100_pyramid200.yaml --expname MixUp --mode CutOut --tensorboard
```

## Transform

```bash
$ python plot.py
```

## Model

#### Baseline

Link：https://pan.baidu.com/s/1kISzxj8J4QnXT7ssDbvOKQ 
Key：zawv 

#### CutOut

Link：https://pan.baidu.com/s/1qCTF-52mn2D1CS1My7W37w 
Key：3dan 

#### MixUp

Link：https://pan.baidu.com/s/1O_Z2Z3cbBgmmUgZ3nl30Qw 
Key：jjdd

#### CutMix

Link：https://pan.baidu.com/s/1ZiV4EfE0WdKKyZ8yjoWSbw 
Key：xbge 
