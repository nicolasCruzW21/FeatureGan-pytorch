set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch
conda install visdom dominate -c conda-forge # install visdom and dominate
