This contains all codes for replicating every results in my Ph. D. thesis.

# Requirements
- Python 3.5 (conda 3.18.6 with python 3.5.0 is recommended)
    - Chainer 1.5.0.2
    - Cython 0.23.4
    - NumPy 1.10.1
    - tqdm
- OpenCV 3.0.0
- lmdb 0.87
- Boost 1.59.0
- Boost.NumPy ([26aaa5b](https://github.com/ndarray/Boost.NumPy/tree/26aaa5b62e6170f2ccde179b46f1a49c4011fc9d))

# Build Libraries
## OpenCV 3.0.0

```
$ wget https://github.com/Itseez/opencv/archive/3.0.0.zip
$ unzip 3.0.0.zip && rm -rf 3.0.0.zip
$ cd opencv-3.0.0 && mkdir build && cd build
$ bash $SSAI_HOME/shells/build_opencv.sh
$ make -j32 install
```

If some libraries are missing, do below before compiling 3.0.0.

```
$ sudo apt-get install -y libopencv-dev libtbb-dev
```

## Boost 1.59. 0

```
$ wget http://downloads.sourceforge.net/project/boost/boost/1.59.0/boost_1_59_0.tar.bz2
$ tar xvf boost_1_59_0.tar.bz2 && rm -rf boost_1_59_0.tar.bz2
$ cd boost_1_59_0
$ ./bootstrap.sh
$ ./b2 -j32 install cxxflags="-I/home/ubuntu/anaconda3/include/python3.5m"
```

## Boost.NumPy

```
$ git clone https://github.com/ndarray/Boost.NumPy.git
$ cd Boost.NumPy && mkdir build && cd build
$ cmake -DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.5m.so ../
$ make install
```

## Build utils

```
$ cd $SSAI_HOME/scripts/utils
$ bash build.sh
```

# Create Dataset

```
$ bash shells/download.sh
$ bash shells/create_dataset.sh
```

Dataset         | Training | Validation | Test
:-------------: | :------: | :--------: | :----:
mass_roads      | 8580352  | 108416     | 379456
mass_roads_mini | 1060928  | 30976      | 77440
mass_buildings  | 1060928  | 30976      | 77440
mass_merged     | 1060928  | 30976      | 77440

# Start Training

```
$ bash shells/train_batch.sh
```

# MnihCNN_multi

[ 0.98854691  0.98864113]
[ 0.95177701  0.95246911]
[ 0.9106115   0.90921618]

# x0_sigma

```
159.348674296
```
