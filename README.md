This contains all codes for replicating every results in my Ph. D. thesis.

# Requirements
- Python 3.5+
- Chainer 1.4.1+
- NumPy 1.10+
- OpenCV 3.0+
- lmdb 0.87+

# Create Dataset

```
$ bash shells/download.sh
$ bash shells/create_dataset.sh
```

Dataset         | Training | Validation | Test
:-------------: | :------: | :--------: | :----:
mass_roads      | 8458173  | 126281     | 440932
mass_roads_mini | 1119872  | 36100      | 89968
mass_buildings  | 1119872  | 36100      | 89968
mass_merged     | 1119872  | 36100      | 89968

# Start Training

```
$ bash shells/train_batch.sh
```
