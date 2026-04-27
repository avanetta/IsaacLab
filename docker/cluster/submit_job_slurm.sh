#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`
module load eth_proxy

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=pro_6000:1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=120g
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=avanetta@ethz.ch
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
