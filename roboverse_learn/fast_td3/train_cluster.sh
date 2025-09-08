#!/bin/bash
#SBATCH --job-name=Stand_train_ppo
#SBATCH --account OPEN-32-27
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
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

# Read parameters
CONFIG_NAME=$1
SIM_NAME=$2   # replaces AG
ALGO_NAME=$3  # replaces ppo
#ROBOT_NAME=$4

# Update job name dynamically using scontrol (works only after submission)
scontrol update JobId=$SLURM_JOB_ID JobName=${SIM_NAME}_${ALGO_NAME}


if [ -z "$SIM_NAME" ] || [ -z "$ALGO_NAME" ]; then
    echo "Usage: sbatch script.sh <ENV_NAME> <ALGO_NAME>"
    exit 1
fi

echo "Running training for environment: $SIM_NAME with algorithm: $ALGO_NAME"



python train.py "$SIM_NAME"


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
