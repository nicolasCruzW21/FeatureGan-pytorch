
- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).


  - For Conda users, create an environment and then run the installation script `./scripts/conda_deps.sh`


The code for this model is based on CycleGan which was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/).


Relevant files are:
feature_gan_model.py
networks.py


To train:
First download the VGG-19 model from the following link:

"https://drive.google.com/file/d/1pjWYa-5_PCpwlNNsa5Rfbb8am7k6vF2I/view?usp=sharing"

Then paste it in the main folder:

To start training:

"python train.py --dataroot ./datasets/SimRobot --name SimRobot --model feature_gan --netG unet_512 --load_size 580 --crop_size 512 --lr 0.00005 --n_epochs 200 --n_epochs_decay 150 --print_freq 10 --display_freq 10"

To infer and trace:

Use your own weights or download a pre-trained model from the following link:

"https://drive.google.com/file/d/1wTNH7gB3ujb3AwebK8AarIoiD7GmrORY/view?usp=sharing"

place the weights file on:
"checkpoints/SimRobot"

Then:

"python test.py --dataroot ./datasets/SimRobot --name SimRobot --model feature_gan --netG unet_512 --num_test 10 --load_size 512 --crop_size 512"

The generated images are placed in the folder "results", the traced model "traced_unet_512" is on the main folder.

