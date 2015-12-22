PYTHON_DIR=$HOME/.pyenv/versions/anaconda3-2.4.0
cmake \
-DPYTHON_LIBRARY=$PYTHON_DIR/lib/libpython3.5m.so \
-DPYTHON_INCLUDE_DIR=$PYTHON_DIR/include/python3.5m \
-DPYTHON_INCLUDE_DIR2=$PYTHON_DIR/include \
-DOpenCV_DIR=/usr/local/share/OpenCV \
../ && make
