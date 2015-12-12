#! /bin/bash

if [ ! -d build ]; then
	mkdir build
fi

cd build

PYTHONDIR=$HOME/.pyenv/versions/anaconda3-2.4.0

cmake \
-DPYTHON_LIBRARY=$PYTHONDIR/lib/libpython3.5m.so \
-DPYTHON_INCLUDE_DIR=$PYTHONDIR/include/python3.5m \
-DOpenCV_DIR=/usr/local/share/OpenCV \
-DPYTHON_INCLUDE_DIR2=$PYTHONDIR/include ../
