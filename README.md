# Face-Recognition-Pytorch

This project is an implementation in PyTorch Lightning and Hydra of the paper [FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR 2015.](https://arxiv.org/pdf/1503.03832.pdf)

## Project Installation

Install the following packages:
```
! pip install wandb
! pip install colorlog
! pip install -U rich
```
Then inside the project, install the dependencies:
```
cd face-recognition-pytorch/
pip install -e .
```

We use [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) for the training and the evaluation of the model.

## How to train the model

Remarks: The parsing is done by using the powerful [Hydra](https://github.com/facebookresearch/hydra) library. You can override anything in the configuration by passing arguments like `foo=value` or `foo.bar=value`.
```
HYDRA_FULL_ERROR=1 python facenet/train.py [OPTIONS]
```
The parameters are specified in `configs/` but can be overridden with the command line

## Face Recognition Prediction
