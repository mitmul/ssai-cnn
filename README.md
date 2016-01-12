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

# Results
## Conventional methods

Model                         | Mass. Buildings | Mass. Roads
:---------------------------- | :-------------- | :----------
MnihCNN                       | 0.9150          | 0.8873
MnihCNN + CRF                 | 0.9211          | 0.8904
MnihCNN + Post-processing net | 0.9203          | 0.9006

## Step = 256

Model                              | Building-channel | Road-channel | epoch
:--------------------------------- | :--------------- | :----------- | :----
ours (multi-channel with MA)       | 0.95132348       | 0.8996919    | 150
ours (multi-channel with CIS + MA) | 0.95229794       | 0.90027631   | 150
ours (multi-channel with MA)       | 0.9516143        | 0.90166159   | 200
ours (multi-channel with CIS)      | 0.940134         | 0.890475     | 200
ours (multi-channel with CIS + MA) | 0.95276838       | 0.90142274   | 200

## Step = 1024

Model                              | Building-channel | Road-channel | epoch
:--------------------------------- | :--------------- | :----------- | :----
ours (multi-channel with MA)       | 0.95152357       | 0.8996919    | 150
ours (multi-channel with CIS + MA) | 0.95183038       | 0.90027631   | 150
ours (multi-channel with MA)       | 0.95144516       | 0.90166159   | 200
ours (multi-channel with CIS + MA) | 0.95298734       | 0.90104864   | 200

# x0_sigma for inverting feature maps

```
159.348674296
```

# After prediction for single MA

```
$ bash shells/predict.sh
$ python scripts/integrate.py --result_dir results --epoch 400 --size 7,60
$ PYTHONPATH=".":$PYTHONPATH python scripts/evaluate.py --map_dir data/mass_merged/test/map --result_dir results/integrated_150 --channel 3 --offset 8 --relax 3 --steps 1024
```
