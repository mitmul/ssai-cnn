#! /bin/bash

SSAI_HOME=$PWD
cd $HOME

if [ ! -d lib ]; then
    mkdir lib
fi
cd lib

pip uninstall -y chainer
if [ ! -d chainer ]; then
    git clone https://github.com/pfnet/chainer.git
fi
cd chainer
git pull
python setup.py install

cd $HOME/lib

if [ ! -d opencv-3.0.0 ]; then
    wget https://github.com/Itseez/opencv/archive/3.0.0.zip
    unzip 3.0.0.zip && rm -rf 3.0.0.zip
fi
cd opencv-3.0.0
if [ ! -d build ]; then
    mkdir build
fi
cd build
setopt RM_STAR_SILENT
rm -rf *
setopt RM_STAR_WAIT
bash $SSAI_HOME/shells/build_opencv.sh
make -j32 install

cd $HOME/lib

if [ ! -d boost_1_59_0 ]; then
    wget http://downloads.sourceforge.net/project/boost/boost/1.59.0/boost_1_59_0.tar.bz2
    tar xvf boost_1_59_0.tar.bz2 && rm -rf boost_1_59_0.tar.bz2
fi
cd boost_1_59_0
./bootstrap.sh
./b2 -j32 install cxxflags="-I${HOME}/anaconda3/include/python3.5m"

cd $HOME/lib

rm -rf Boost.NumPy
git clone https://github.com/ndarray/Boost.NumPy.git
cd Boost.NumPy && mkdir build && cd build
cmake -DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.5m.so ../
make -j32 install

cd $SSAI_HOME/scripts/utils
python setup.py build_ext -i

cd $SSAI_HOME/scripts/utils/transform
cmake \
-DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.5m.so \
-DPYTHON_INCLUDE_DIR=$HOME/anaconda3/include \
-DPYTHON_INCLUDE_DIR2=$HOME/anaconda3/include/python3.5m \
. && make

cd $SSAI_HOME

bash shells/download.sh
bash shells/create_dataset.sh

bash shells/train_batch.sh
