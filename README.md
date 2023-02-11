# FedLP:  Layer-wise Pruning Mechanism for Communication-Computation Efficient Federated Learning

This is the implemention of FedLP, a Federated learning (FL) framework with Layer-wise Pruning. (Accepted by ICC 2023)

## Requirements

python>=3.7  
pytorch>=1.12.0

## Main Files

Centralized model training is performed by
> [main_nn.py](main_nn.py)

Federated learning (vanilla FedAvg) is performed by:
> [main_fed.py](main_fed.py)

Heterogenous FedLP is performed by:
> [main_fed_hetero.py](main_fed_hetero.py)

Homogenous FedLP (with upload dropout) is performed by:
> [main_fed_cplx.py](main_fed_cplx.py)

Homogenous FedLP (with upload and download dropout) is performed by:
> main_fed_drop.py *(not finished)

## Other files
### ./utils
See the arguments in [options.py](utils/options.py). 

Get datasets and the distribution of datasets in [sampling.py](utils/sampling.py).

### ./models
Uploading process (FedAvg-based methods) in [Fed.py](models/Fed.py)
* ```FedAvg()``` for vanilla FedAvg
* ```fed_hetero()``` for heterogenous case
* ```fed_dropout()``` for homogenous case (with upload dropout)

Dataset spliting, Local updating and Download dropout in [Update.py](models/Update.py)
* ```download_sampling``` for homogenous case (downloading prosess)...
### 


## Results
We develop the experiments in an image classification FL
task under CIFAR-10 dataset. We build up a FL system with 100 clients in total. The participation rate is set as 0.1, local epoch is set as 5 and communication
round as 200.

| Scheme    | Test accuracy         | Comm.     | Comp. |
| -----     | -----                 | ----      | ----  |
|           | iid / D-niid / M-niid | #param(k) | MFLOPs|
| FedAvg    | 77.94 / 77.67 / 67.57 | 1102.93   | 36.36 |
| FedLP-Homo(0.1) | 75.32 / 71.30 / 44.21 | 606.61 | 36.36 |
| FedLP-Homo(0.3) | 78.20 / 74.92 / 63.24 | 716.91 | 36.36 |
| FedLP-Homo(0.5) | 77.60 / 77.13 / 66.01 | 827.20 | 36.36 |
| FedLP-Homo(0.7) | 78.47 / 77.71 / 70.29 | 937.49 | 36.36 |
| FedLP-Hetero(1) | 66.00 / 67.66 / 37.30 | 169.60 | 17.73 |
| FedLP-Hetero(3) | 68.82 / 68.29 / 39.54 | 225.28 | 24.62 |
| FedLP-Hetero(u) | 72.42 / 64.65 / 57.51 | 318.66 | 23.83 |
| FedLP-Hetero(5) | 76.28 / 76.34 / 65.69 | 710.80 | 30.10 |




