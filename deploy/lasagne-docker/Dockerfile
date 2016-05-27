FROM  nvidia/cuda:7.5-cudnn5-devel


# Get dependencies


RUN apt-get update && apt-get install -y \
bc \
cmake \
curl \
gfortran \
git \
libprotobuf-dev \
libleveldb-dev \
libsnappy-dev \
libopencv-dev \
libboost-all-dev \
libhdf5-serial-dev \
liblmdb-dev \  
libjpeg62 \
libfreeimage-dev \
libatlas-base-dev \
libgflags-dev \
libgoogle-glog-dev \
pkgconf \
protobuf-compiler \
python-dev \
python-pip \
python-opencv \
python-numpy \
unzip \
wget \
vim \
htop \
sshfs \
cifs-utils \
tmux



# Allow it to find CUDA libs
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
ldconfig

# Instal pip packages used by various scripts, and useful for interactive dev
RUN sudo pip install pydicom lmdb jupyter plyvel peewee nibabel tqdm pypng natsort medpy psutil
RUN pip install --allow-insecure www.simpleitk.org -f http://www.simpleitk.org/SimpleITK/resources/software.html --timeout 30 SimpleITK
# Install NLopt
RUN cd /opt/ && wget http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz && tar -xvf nlopt-2.4.2.tar.gz && rm nlopt-2.4.2.tar.gz
RUN cd /opt/nlopt-2.4.2 && ./configure --enable-shared && make -j$(nproc) && make install
RUN echo "/opt/nlopt-2.4.2/.libs/" > /etc/ld.so.conf.d/nlopt.conf && ldconfig

RUN pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

#configure theano to use GPU by default
RUN echo "[global] \ndevice = gpu0" >> .theanorc


EXPOSE 8888 8889 8890 8891 8892 8893 8894 8895 8896 8897 8898 8899 8900


