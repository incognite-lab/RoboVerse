#!/bin/bash
#SBATCH --job-name=Stand_train_ppo
#SBATCH --account OPEN-32-27
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --time=24:00:00
#SBATCH --partition=qcpu
# Define and create a unique scratch directory for this job
# /lscratch is local ssd disk on particular node which is faster
# than your network home dir
#SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.FM_tiago_rotslide_m2
#mkdir -p ${SCRATCH_DIRECTORY}
#cd ${SCRATCH_DIRECTORY}


# Načtení modulů pro GPU
module purge
module load foss/2023a
module load CUDA/11.7.0
module load cudatoolkit
module load mesa

# Aktivace Conda env (plná cesta)
source /scratch/project/open-32-27/miniconda3/etc/profile.d/conda.sh
conda activate /home/zemlifi1/.conda/envs/metasim

# Headless MuJoCo
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
#export MUJOCO_GL=egl

# Update job name
scontrol update JobId=$SLURM_JOB_ID JobName=mujoco_ppo

# Spuštění
python get_started/0_static_scene.py --sim sapien3 --headless
