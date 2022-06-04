#!/bin/bash
#
source /home/cass/anaconda3/bin/activate
conda activate spectra

data="AmazDigiMu_full"

shopt -s nullglob

# python -m rationalizers train --config configs/cus/seed25.yaml

for i in $(find ./experiments/$data/ -type f -iname "*.ckpt"); do
echo $i
python -m rationalizers predict --config configs/cus/seed25.yaml --ckpt $i
done