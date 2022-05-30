#!/bin/bash
#
source /home/cass/anaconda3/bin/activate
conda activate spectra

data="factcheck"

shopt -s nullglob


for i in $(find ./experiments/$data/ -type f -iname "*.ckpt"); do
echo $i
python -m rationalizers predict --config configs/factcheck/factcheck_spectra.yaml --ckpt $i
done