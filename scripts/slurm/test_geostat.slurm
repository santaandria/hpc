#!/bin/sh
#SBATCH --job-name test_geostat
#SBATCH --error test_geostat.%j.err
#SBATCH --output test_geostat.%j.out
##SBATCH --mail-user f.santaandrea@gmail.com
##SBATCH --mail-type END,FAIL
##SBATCH --partition allgroups
#SBATCH --partition LocalQ
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 01:00

cd $SLURM_SUBMIT_DIR
srun singularity exec geostat.sif python test.py