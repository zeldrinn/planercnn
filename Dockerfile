FROM tobycheese/cuda:9.0-cudnn7-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	tmux \
	nano \
        apt-utils \
        python3.6 \
        python3.6-dev \
        python3-pip \
	python3-wheel \
        python3-setuptools \
        g++-6 \
        libsm6 \
        libxrender1 \
        libxtst6 \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update
    
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10 && \
    update-alternatives --set g++ /usr/bin/g++-6
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10 && \
    update-alternatives --set gcc /usr/bin/gcc-6

RUN git clone -b dockerfile https://github.com/zeldrinn/planercnn.git

RUN cd planercnn && python3 -m pip install --upgrade pip && pip3 install wheel && pip3 install h5py torch==0.4.1 -f https://download.pytorch.org/whl/cu92/torch_stable.html && pip3 install -r requirements.txt 

RUN gcc -v && cat /usr/local/cuda/version.txt

RUN ls /usr/local | grep cuda

RUN cd planercnn && \
    cd nms/src/cuda/ && \
    nvcc -c -o nms_kernel.cu.o nms_kernel.cu -I /usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_37
RUN export LD_LIBRARY_PATH="/usr/local/cuda/include:$LD_LIBRARY_PATH" && \
    cd planercnn/nms && \
    sed -i 's|extra_objects=extra_objects|extra_objects=extra_objects,\n    include_dirs=["/usr/local/cuda/include"]|' build.py && \
    python3 build.py
RUN cd planercnn/roialign/roi_align/src/cuda && \
    nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -I /usr/local/cuda/include -x cu -Xcompiler -fPIC -arch=sm_37
RUN cd planercnn/roialign/roi_align && \
    export LD_LIBRARY_PATH="/usr/local/cuda-9.2/include:$LD_LIBRARY_PATH" && \
    sed -i 's|extra_compile_args=extra_compile_args|extra_compile_args=extra_compile_args,\n    include_dirs=["/usr/local/cuda/include"]|' build.py && \
    python3 build.py

RUN pip3 install torch==0.4.1 -f https://download.pytorch.org/whl/cu92/torch_stable.html
