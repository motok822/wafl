
#!/bin/bash

#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -W group_list=gn77
#PBS -j oe

module load cuda/12.4
cd n77002/wafl/WAFL-DIAL
source .venv/bin/activate
which python
python src/main.py