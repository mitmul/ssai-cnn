#! /bin/bash
export PYTHONPATH=".":$PYTHONPATH

python scripts/create_dataset.py --dataset multi
python scripts/create_dataset.py --dataset single
python scripts/create_dataset.py --dataset roads_mini
python scripts/create_dataset.py --dataset roads
python scripts/create_dataset.py --dataset buildings
python scripts/create_dataset.py --dataset merged

bash shells/test_dataset.sh
bash shells/test_transform.sh
