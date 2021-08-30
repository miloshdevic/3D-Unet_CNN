#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH ntasks-per-node=4
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-user=milosh.devic@gmail.com
#SBATCH mail-type=ALL

cd $project/def-enger/mdevic31/
module purge # ?
module load python/3.8.10 scipy-stack
source ~/ENV/bin/activate

python model_3D-Unet.py

