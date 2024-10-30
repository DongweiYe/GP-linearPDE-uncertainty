#!/bin/bash
#SBATCH --job-name=plot_h
#SBATCH -p mia,mia-pof,am
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=200G
#SBATCH --gres=gpu:1
#SBATCH --time=64:03:00
#SBATCH --output=plot_h_%j.log
#SBATCH --error=err_%j.log
#SBATCH --exclude=hpc-node05,hpc-node06


hostname
start_time=$(date +%s)
cd /local
mkdir ${SLURM_JOBID}

ScratchDir="/local/${SLURM_JOBID}"
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' already found !"
else
   echo "'$ScratchDir' not found, creating !"
   mkdir $ScratchDir
fi
cd $ScratchDir


#cd ${SLURM_JOBID}
cp -r ${SLURM_SUBMIT_DIR}/* .
export OMPI_MCA_mca_base_component_show_load_errors=0
module load nvidia/cuda-11.8
module load nvidia/cuda-11.x_cudnn-8.6
module load nvidia/nvhpc/23.3
module load nvidia/nvtop

#pip install -U jax[cuda11_cudnn86] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# -----------------Print node info--------------------------------
nvidia-smi            # log GPU info
echo "Date              = $(date)"
today=$(date +%m%d)
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of MPI tasks         : "$SLURM_NTASKS
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI tasks per node: "$SLURM_TASKS_PER_NODE
echo "Number of threads per core  : "$SLURM_THREADS_PER_CORE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "
# -----------------end--------------------------------



# -----------------run python---------------------
export JAX_PLATFORM_NAME=gpu
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mpirun -np 1 python main_plot_h.py
# -----------------end--------------------------------



# -----------------print versions---------------------
packages=("numpy" "jaxopt" "scipy" "jax" "jaxlib" "jaxopt" )

for package in "${packages[@]}"; do
  echo "${package} version:" >> ${SLURM_SUBMIT_DIR}/results/log/${today}/${SLURM_JOBID}_version.log
  pip list | grep "${package}" >> ${SLURM_SUBMIT_DIR}/results/log/${today}/${SLURM_JOBID}_version.log
done
# -----------------end--------------------------------


# -----------------print excute time-----------------
end_time=$(date +%s)
running_time=$((end_time - start_time))
echo "Script finished in $((running_time / 60)) minutes and $((running_time % 60)) seconds."
echo "Script finished in $((running_time / 60)) minutes and $((running_time % 60)) seconds." >> ${SLURM_SUBMIT_DIR}/results/log/${today}/${SLURM_JOBID}_version.log
# -----------------end--------------------------------



# -----------------get output---------------------
mkdir -p ${SLURM_SUBMIT_DIR}/results/figures/${today}
mkdir -p ${SLURM_SUBMIT_DIR}/results/log/${today}
mkdir -p ${SLURM_SUBMIT_DIR}/results/datas/trained_params/${today}

cp *.pdf ${SLURM_SUBMIT_DIR}/results/figures/${today}
cp *.png ${SLURM_SUBMIT_DIR}/results/figures/${today}
cp *.pkl ${SLURM_SUBMIT_DIR}/results/datas/trained_params/${today}

cd ${SLURM_SUBMIT_DIR}
mv *.log ${SLURM_SUBMIT_DIR}/results/log/${today}

#cp ${SCRATCH_DIRECTORY} ${SLURM_SUBMIT_DIR}
#echo "SCRATCH_DIRECTORY is set to: ${SCRATCH_DIRECTORY}"
##export SCRATCH_DIRECTORY=/local/${SLURM_JOBID}
#rm -rf ${SCRATCH_DIRECTORY}

# Clean up on the compute node !
cd ~
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' found and now copying files, please wait ..."
   rm -rf $ScratchDir
else
   echo "Warning: '$ScratchDir' NOT found."
fi

# -----------------end--------------------------------
# Finish the script
exit 0