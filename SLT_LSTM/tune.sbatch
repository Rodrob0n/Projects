#!/bin/bash
#SBATCH --job-name=LSTMTuning           # Job name
#SBATCH --partition=falcon,gecko         # Partition(s) to use
#SBATCH --ntasks=1                      # One task
#SBATCH --cpus-per-task=12              # Number of CPU threads
#SBATCH --gres=gpu:1                    # Request one GPU
#SBATCH --time=24:00:00                 # Maximum run time of 12 hours
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80  # Email notifications
#SBATCH --output=tune_%j.out       # Standard output file
#SBATCH --error=tune_%j.err        # Standard error file

# Load system modules
source /etc/profile.d/modules.sh
module load CUDA/11.6.0
module load python-ml/22-12-21-python3.9

# Set CUDA environment variables (adjust paths if necessary)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/local/java/cuda-11.6.0"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/local/java/cuda-11.6.0/lib64

# Verify that the GPU is visible
nvidia-smi

# Install dependencies in --user mode
python3.9 -m pip install --user --upgrade protobuf==3.19.4
# If needed, install a specific version of TensorFlow:
python3.9 -m pip install --user tensorflow==2.18.0
python3.9 -m pip install --user keras-tuner


# Run your training script
#python3.9 train5_unpadded.py

python3.9 fineTuning.py
