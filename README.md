# A baseline for MC_dropout calibration
### Training Lenet model for CIFAR-10
running: python train.py 
### Then get calibrated model using post-processing method([temperature scaling](https://github.com/gpleiss/temperature_scaling)) 
running: python calibrated_model.py 
### Prerequisites
* PyTorch
* Numpy
* Matplotlib

The project is written in python 3.6 and Pytorch 1.4.0. If CUDA is available, it will be
used automatically. The models can also run on CPU as they are not excessively big.
