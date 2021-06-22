#ARG PYTORCH="1.3"
#ARG CUDA="10.1"
#ARG CUDNN="7"
#
#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#
#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
#
#RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
# && apt-get clean \
# && rm -rf /var/lib/apt/lists/*
#
## Install mmdetection
#RUN conda clean --all
#RUN git clone https://zengzhaoyang:caD7rovL@github.com/zengzhaoyang/iqiyi1.git /mmdetection
#WORKDIR /mmdetection
#ENV FORCE_CUDA="1"
#RUN pip install --no-cache-dir -e .
#RUN pip install cython
#RUN pip install pycocotools opencv-python
#RUN pip install pillow==6.2.2
#RUN pip install numpy==1.17
FROM zengzhaoyang/iq:v1
RUN cd /mmdetection && git pull
