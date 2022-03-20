#!/bin/bash
#SBATCH --time=1-23:00:00
#SBATCH --account=def-zdziong
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=mohamad-arfah.dabberni.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL
#SBATCH --output=job_main.out

module load python/3.6
source .venv/bin/activate

python ./apps/genetic_selectors_v2/main.py
