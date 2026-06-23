#!/bin/bash
#SBATCH --job-name=prop_rl
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH -A marlowe-m000243
#SBATCH --time=04:00:00
#SBATCH --output=/projects/m000243/nazirk/voting/logs/rl_%j.out
#SBATCH --error=/projects/m000243/nazirk/voting/logs/rl_%j.err

module load python3
module load cudnn/cuda12/9.3.0.75

export HF_HOME=/projects/m000243/nazirk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /projects/m000243/nazirk/voting/clean

pip install torch numpy transformers accelerate bitsandbytes pandas tqdm \
    sentence-transformers nltk langdetect --user -q

python3 rl.py \
    --max_iters 20 \
    --sim_threshold 0.85 \
    --ppl_factor 2.0
