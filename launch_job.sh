#!/bin/bash

#SBATCH -c 20
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH -o /home/gridsan/vparimi/info-mapf/runs/job.log-%j

source /etc/profile
module unload anaconda
module load anaconda/2023a-pytorch

source activate info-mapf

project_root=/home/gridsan/vparimi/info-mapf
unset PYTHONPATH
export PYTHONPATH=$project_root:$PYTHONPATH

python -u test/test_real_world_setup.py --dataset_name galveston-bay.xyz --cell_size_degrees 0.001 --logging_level debug
