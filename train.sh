#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=xfact-spectra

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --partition=small

#SBATCH --mem=60GB

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk





# run the application
cd /jmain02/home/J2AD003/txk58/zxz22-txk58/spectra/spectra-rationalization
module load python/anaconda3
module load cuda/10.2
source activate spectra

python -m rationalizers train --config configs/hotels/hotels_spectra.yaml