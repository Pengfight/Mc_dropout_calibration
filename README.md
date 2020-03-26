# A baseline for MC_dropout calibration
## Lenet model
### Training Lenet model for CIFAR-10
running: python train.py 
### Then get calibrated model using post-processing method([temperature scaling](https://github.com/gpleiss/temperature_scaling)) 
running: python calibrated_model.py 
## Densenet model
### train a DenseNet on CIFAR100, and save the validation indices:
```sh
python train_densenet.py
```
a densenet40 can be used directly in 'model' directory(model.pth)
### Then get calibrated model
```sh
python caibrated_densenet.py --data <path_to_data> --save <save_folder_dest>
```
a calibrated densenet40 can be used directly in 'model' directory(model_with_temperature.pth)
### Prerequisites
* PyTorch
* Numpy
* Matplotlib

The project is written in python 3.6 and Pytorch 1.4.0. If CUDA is available, it will be
used automatically. The models can also run on CPU as they are not excessively big.
