# From local to server 
# scp Dataset.zip ${USER}@ssh.inf.ed.ac.uk:/afs/inf.ed.ac.uk/user/s20/${USER}
# From dice to server 
# scp ${USER}@ssh.inf.ed.ac.uk:/afs/inf.ed.ac.uk/user/s20/${USER}/Dataset.zip ${dest_path}

srun --time=24:00:00 --mem=16000 --cpus-per-task=8 --gres=gpu:1 --partition=PGR-Standard --pty bash

sbatch -N 1 -n 1 --mem=16000 \
# --nodelist=damnii10 \ # you can assigin task with anynode
--time 12:00:00 \ # adjust your training time with estimation
--cpus-per-task=4 \
--gres=gpu:1 \
--job-name=? \ #fill up here
--partition=PGR-Standard \
--output=/home/%u/slurm_logs/slurm-%A_%a.out \ # This will pass the info logs to this path
--error=/home/%u/slurm_logs/slurm-%A_%a.out \ # This will pass the error logs to this path

echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"
echo "Setting up bash enviroment"
source ~/.bashrc
set -e
SCRATCH_DISK=/disk/scratch_big
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}
CONDA_ENV_NAME= ? # Fill up here
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
repo_home=/home/${USER}/? # Fill up here
src_path=/home/${USER}/? # Fill up here
dest_path=${SCRATCH_HOME}/? # Fill up here
mkdir -p ${dest_path}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}
echo "Start train"

python train.py --batch 32 --epochs 100 --data data/${Dataset}.yaml --weights weights/yolov5s.pt --workers 20

# Post experiment logging
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"