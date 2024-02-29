#!/bin/sh
#SBATCH --job-name geostat
#SBATCH --error geostat.%j.err
#SBATCH --output geostat.%j.out
##SBATCH --mail-user f.santaandrea@gmail.com
##SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntask 2
#SBATCH --mem 8G
#SBATCH --time 30:00

spack load openmpi@3.1.4

SINGULARITY_IMAGE="scripts/singularity/images/geostat.sif"
cd $SLURM_SUBMIT_DIR/src
#srun singularity exec $SINGULARITY_IMAGE dask-mpi --local-directory /tmp/$USER/dask --scheduler-file $SLURM_SUBMIT_DIR/scheduler.json &
mpirun -n $SLURM_NTASKS singularity exec $SINGULARITY_IMAGE dask-mpi --local-directory /tmp/$USER/dask --scheduler-file $SLURM_SUBMIT_DIR/scheduler.json &
sleep 10
singularity exec $SINGULARITY_IMAGE python main.py --scheduler-file $SLURM_SUBMIT_DIR/scheduler.json
rm -f $SLURM_SUBMIT_DIR/scheduler.json
