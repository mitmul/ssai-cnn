#! /bin/bash

SSAI_HOME=$PWD
cd $HOME

if [ ! -d lib ]; then
    mkdir lib
fi
cd lib

wget https://github.com/Itseez/opencv/archive/3.0.0.zip
unzip 3.0.0.zip && rm -rf 3.0.0.zip
cd opencv-3.0.0 && mkdir build && cd build
bash $SSAI_HOME/shells/build_opencv.sh
make -j32 install

wget http://downloads.sourceforge.net/project/boost/boost/1.59.0/boost_1_59_0.tar.bz2
tar xvf boost_1_59_0.tar.bz2 && rm -rf boost_1_59_0.tar.bz2
cd boost_1_59_0
./bootstrap.sh
./b2 -j32 install cxxflags="-I/home/ubuntu/anaconda3/include/python3.5m"

git clone https://github.com/ndarray/Boost.NumPy.git
cd Boost.NumPy && mkdir build && cd build
cmake -DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.5m.so ../
make install

cd $SSAI_HOME/scripts/utils
python setup.py build_ext -i

cd $SSAI_HOME/scripts/utils/transform
cmake \
-DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.5m.so \
-DPYTHON_INCLUDE_DIR=$HOME/anaconda3/include/python3.5m \
. && make

bash shells/download.sh
bash shells/create_dataset.sh

bash shells/train_batch.sh
