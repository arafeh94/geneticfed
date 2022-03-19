#!/bin/bash
#SBATCH --time=1-23:00:00
#SBATCH --account=def-zdziong
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=ahmad.hammoud.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL
#SBATCH --output=job_main.out

source  ./venv/bin/activate
python ./apps/experiments/fedavg