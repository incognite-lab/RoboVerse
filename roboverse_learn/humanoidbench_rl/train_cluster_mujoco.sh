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


# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was
# submitted from (usually in your network home dir)
#cp -r ${SLURM_SUBMIT_DIR}/myGym/myGym/ ${SCRATCH_DIRECTORY}
source /scratch/project/open-32-27/miniconda3/etc/profile.d/conda.sh
conda activate metasim



# Update job name dynamically using scontrol (works only after submission)
scontrol update JobId=$SLURM_JOB_ID JobName=mujoco_ppo

python train_sb3.py mujoco


# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
#cp -r ${SCRATCH_DIRECTORY} ${SLURM_SUBMIT_DIR}/output

# In addition to the copied files, you will also find a file called
# slurm-1234.out in the submit directory. This file will contain all output that
# was produced during runtime, i.e. stdout and stderr.

# After everything is saved to the home directory, delete the work directory to
# save space on /lscratch
# old files in /lscratch will be deleted automatically after some time
#cd ${SLURM_SUBMIT_DIR}
#srm -rf ${SCRATCH_DIRECTORY}
