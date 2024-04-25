FROM nvcr.io/nvidia/pytorch:22.05-py3

ENV DEBIAN_FRONTEND noninteractive

RUN export TORCH_CUDA_ARCH_LIST=8.6

RUN apt update && apt install -y tmux nano
RUN apt-get update && apt-get install -y libsparsehash-dev python-yaml
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN pip install easydict pyquaternion
RUN pip install nuscenes-devkit

RUN pip install pytorch_lightning==1.7.1
RUN pip install spconv-cu116==2.2.4
RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu117.html

# fix some minor version mismatch
RUN pip uninstall -y  opencv-python
RUN pip install opencv-python==4.5.5.64
RUN pip install torchmetrics==0.11.4