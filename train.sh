#!/bin/bash
#
source /home/cass/anaconda3/bin/activate
conda activate spectra



python -m rationalizers train --config configs/cus/seed5.yaml
python -m rationalizers train --config configs/cus/seed10.yaml
python -m rationalizers train --config configs/cus/seed15.yaml
python -m rationalizers train --config configs/cus/seed20.yaml
python -m rationalizers train --config configs/cus/seed25.yaml
