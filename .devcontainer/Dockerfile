FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update
RUN apt -y update

##
RUN apt -y install curl wget vim
# RUN apt -y install python3.9
RUN apt -y install ffmpeg
RUN echo "alias python=python3.8" >> ~/.bashrc
RUN echo "alias python3=python3.8" >> ~/.bashrc
RUN source ~/.bashrc
RUN apt -y install python3-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

##
RUN apt -y install libopencv-dev
RUN apt -y install git x11-apps xserver-xorg
RUN apt -y install python3-dev

##
RUN pip3 install -U pip setuptools==57.5.0
RUN pip3 install matplotlib ipython seaborn jupyter
RUN pip3 install Pillow scikit-learn scikit-image
RUN pip3 install tqdm pandas
RUN pip3 install black flake8 prospector mypy bandit vulture dodgy
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torchmetrics captum torchray
RUN pip3 install cmapy ffmpeg-python h5py pyclustering openmim wandb
RUN pip3 install kornia omegaconf einops vit_pytorch
RUN pip3 install openpack-toolkit openpack-torch optuna pymysql hydra-optuna-sweeper
RUN pip3 install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
RUN pip3 install --upgrade requests
RUN pip3 install xgboost

##
RUN mkdir /root/resnet3d_installer
COPY resnet3d_installer/* /root/resnet3d_installer/
RUN chmod 764 /root/resnet3d_installer/get_resnet3d.sh
RUN cd /root \
    && ./resnet3d_installer/get_resnet3d.sh \
    && cp -r resnet3d/resnet3d.egg-info /usr/local/lib/python3.8/dist-packages/ \
    && cp -r resnet3d /usr/local/lib/python3.8/dist-packages/ \
    && rm -r resnet3d \
    && rm -r resnet3d_installer

RUN pip3 install 'git+https://github.com/sooftware/conformer'
RUN pip3 install torchmetrics==0.10.2

WORKDIR /root
