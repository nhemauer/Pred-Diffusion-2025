#!/bin/bash
# launch_all.sh
# Loops over every .py file in scripts/ and submits a job for each

for script in /storage/work/ndh5286/Projects/Pred_Diffusion_2025/ml_covariate_analysis/ml_feature_importance/*.py; do
    name=$(basename "$script" .py)
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=bbd5087_cr_default
#SBATCH --qos=normal
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --mem=384gb
#SBATCH --time=24:00:00
#SBATCH --job-name=${name}
#SBATCH --chdir=/storage/work/ndh5286/Projects/Pred_Diffusion_2025
#SBATCH --output=ml_covariate_analysis/ml_feature_importance/logs/%x_%j.out

python "$script"
EOF

done