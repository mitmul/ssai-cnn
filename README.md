This is an implementation of [Volodymyr Mnih's dissertation](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) methods on his [Massachusetts road & building dataset](https://www.cs.toronto.edu/~vmnih/data/) and my original methods that are published in [this paper](http://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003).

# Requirements

- Python 3.5 (anaconda with python 3.5.1 is recommended)
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

## Boost 1.59\. 0

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

    Dataset     | Training | Validation |  Test
:-------------: | :------: | :--------: | :----:
  mass_roads    | 8580352  |   108416   | 379456
mass_roads_mini | 1060928  |   30976    | 77440
mass_buildings  | 1060928  |   30976    | 77440
  mass_merged   | 1060928  |   30976    | 77440

# Start Training

```
$ CHAINER_TYPE_CHECK=0 CHAINER_SEED=$1 \
nohup python scripts/train.py \
--seed 0 \
--gpu 0 \
--model models/MnihCNN_multi.py \
--train_ortho_db data/mass_merged/lmdb/train_sat \
--train_label_db data/mass_merged/lmdb/train_map \
--valid_ortho_db data/mass_merged/lmdb/valid_sat \
--valid_label_db data/mass_merged/lmdb/valid_map \
--dataset_size 1.0 \
> mnih_multi.log 2>&1 < /dev/null &
```

# Prediction

```
python scripts/predict.py \
--model results/MnihCNN_multi_2016-02-03_03-34-58/MnihCNN_multi.py \
--param results/MnihCNN_multi_2016-02-03_03-34-58/epoch-400.model \
--test_sat_dir data/mass_merged/test/sat \
--channels 3 \
--offset 8 \
--gpu 0 &
```

# Evaluation

```
$ PYTHONPATH=".":$PYTHONPATH python scripts/evaluate.py \
--map_dir data/mass_merged/test/map \
--result_dir results/MnihCNN_multi_2016-02-03_03-34-58/ma_prediction_400 \
--channel 3 \
--offset 8 \
--relax 3 \
--steps 1024
```

# Results

## Conventional methods

Model                         | Mass. Buildings | Mass. Roads            | Mass.Roads-Mini
:---------------------------- | :-------------- | :--------------------- | :--------------
MnihCNN                       | 0.9150          | 0.8873                 | N/A
MnihCNN + CRF                 | 0.9211          | 0.8904                 | N/A
MnihCNN + Post-processing net | 0.9203          | 0.9006                 | N/A
Single-channel                | 0.9503062       | 0.91730195 (epoch 120) | 0.89989258
Single-channel with MA        | 0.953766        | 0.91903522 (epoch 120) | 0.902895

## Multi-channel models (epoch = 400, step = 1024)

Model                       | Building-channel | Road-channel | Road-channel (fixed)
:-------------------------- | :--------------- | :----------- | :-------------------
Multi-channel               | 0.94346856       | 0.89379946   | 0.9033020025
Multi-channel with MA       | 0.95231262       | 0.89971473   | 0.90982972
Multi-channel with CIS      | 0.94417078       | 0.89415726   | 0.9039476538
Multi-channel with CIS + MA | 0.95280431       | 0.90071099   | 0.91108087

## Test on urban areas (epoch = 400, step = 1024)

Model                       | Building-channel | Road-channel
:-------------------------- | :--------------- | :-----------
Single-channel with MA      | 0.962133         | 0.944748
Multi-channel with MA       | 0.962797         | 0.947224
Multi-channel with CIS + MA | 0.964499         | 0.950465

# x0_sigma for inverting feature maps

```
159.348674296
```

# After prediction for single MA

```
$ bash shells/predict.sh
$ python scripts/integrate.py --result_dir results --epoch 200 --size 7,60
$ PYTHONPATH=".":$PYTHONPATH python scripts/evaluate.py --map_dir data/mass_merged/test/map --result_dir results/integrated_200 --channel 3 --offset 8 --relax 3 --steps 256
$ PYTHONPATH="." python scripts/eval_urban.py --result_dir results/integrated_200 --test_map_dir data/mass_merged/test/map --steps 256
```

# Pre-trained models and Predicted results

- [Pre-trained models](https://github.com/mitmul/ssai-cnn/wiki/Pre-trained-models)
- [Predicted results](https://github.com/mitmul/ssai-cnn/wiki/Predicted-results)

# Reference

If you use this code for your project, please cite this journal paper:

- [Multiple Object Extraction from Aerial Imagery with Convolutional Neural Networks](http://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003) ([bibtex](http://www.ingentaconnect.com/content/ist/jist/2016/00000060/00000001/art00003;jsessionid=3bmr095n0lb07.alice?format=bib))

```Shunta Saito, Takayoshi Yamashita, Yoshimitsu Aoki, "Multiple Object Extraction from Aerial Imagery with Convolutional Neural Networks", Journal of Imaging Science and Technology, Vol. 60, No. 1, pp. 10402-1-10402-9, 2015```
