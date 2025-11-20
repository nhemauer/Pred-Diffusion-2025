# Everybody Out of the Pool! A Predictive Assessment of Models of Policy Diffusion in the U.S. States
This repository includes the replication data for Hemauer, Saunders, and Desmarais.

**Abstract:** <br>
How accurately can state policy diffusion be predicted? While prior research has emphasized the use of theory coupled with statistical inference, largely through pooled event history analysis, there has been little evaluation of the predictive performance of these models. Our article benchmarks the predictive accuracy of existing approaches using data from 10 studies of policy diffusion. Alongside traditional logistic regression, we evaluate machine learning methods including XGBoost, random forests, and regularized logistic regression. We assess prediction across four experimental designs: random split, temporal forecasting, policy-specific prediction, and state-specific prediction. Then, we explore the data further by examining logistic regression coefficients over time, and through the use of partial dependence plots. Our results reveal that machine learning methods provide improvements over traditional models, however, it is clear that policy diffusion remains difficult to predict well. Our findings suggest that there remains substantial opportunity to improve policy diffusion theory and models.

**Repository Structure:**
- 'data/': Contains replication data from the original studies we examined in this article.
- 'ml_coef_split/': Contains Stata code to replicate the temporal coefficient plots.
- 'ml_forecast/': Contains Python code to replicate our Forecasting experiment.
- 'ml_hyperparameter/': Contains Python code to prune large hyperparameter grids into smaller grids for efficient model tuning.
- 'ml_policy": Contains Python code to replicate our Policy experiment. 
- 'ml_random": Contains Python code to replicate our Random Split experiment.
- 'ml_simulation/': Contains R code to replicate our simulated data, and Python code to replicate our partial dependence plot analyses.
- 'ml_state/': Contains Python code to replicate our State experiment.

Most repositories include a "bin" folder with a Bash script to run code on a HPC.
Some repositories include a "full_models" folder. These scripts are not meant to be run, but were used to split code into manageable scripts for the HPC.
All repositories include a "figures" folder with generated figures/data from the analyses.

**Note:** <br>
Most of files cannot be run without a HPC. All scripts were made to be run with 48 cores, 336 GB RAM, and 2 weeks of runtime. 