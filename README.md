# Everybody Out of the Pool! A Predictive Assessment of Models of Policy Diffusion in the U.S. States
This repository includes the replication data for Hemauer, Saunders, and Desmarais.

**Abstract:** <br>
How accurately can state policy diffusion be predicted? While prior research has emphasized the use of event history analysis, there has been little evaluation of the predictive performance of these models. Our article benchmarks the predictive accuracy of existing approaches using data from 10 studies of US policy diffusion. Alongside traditional logistic regression, we evaluate machine learning methods including XGBoost, random forests, and regularized logistic regression. We assess prediction across five experimental designs: random split, temporal forecasting, policy-specific prediction, state-specific prediction, and adoption timing. Our analysis provides a baseline assessment of out-of-sample predictive performance, identifies the limitations of current modeling strategies, and highlights the improvements that nonlinear models provide. Our results illustrate how prediction can bolster substantive conclusions in the study of policy diffusion.

**Repository Structure:**
- 'data/': Contains replication data from the original studies we examined in this article.
- 'ml_adoption_timing/': Contains Python code to replicate our Adoption Timing experiment.
- 'ml_covariate_analysis/': Contains Python code to replicate our logistic regression coefficient split, feature importance, and LRT results.
- 'ml_forecast/': Contains Python code to replicate our Forecasting experiment.
- 'ml_hyperparameter/': Contains Python code to prune large hyperparameter grids into smaller grids for efficient model tuning and the best hyperparameters for our random-split experiment.
- 'ml_pdp/': Contains R code to replicate our logistic regression simulated data, and Python code to replicate our partial dependence plot analyses.
- 'ml_policy/": Contains Python code to replicate our Policy experiment. 
- 'ml_random/": Contains Python code to replicate our Random Split experiment.
- 'ml_state/': Contains Python code to replicate our State experiment.

**Folder Structure:**
- The prefix 'ml_' designates folders which include a nested analysis.
- Most folders include a "bin" folder with a Bash script to run code on a HPC.
- Some folders include a "full_models" folder. These scripts are not meant to be run, but were used to split code into manageable scripts for the HPC.
- All folders include a "figures" folder with generated figures/data from the analyses.
- 'ml_state' and 'ml_policy' have a "collapse_all_csv.py" script and a "aggregate_policy.py" script in the figures folder to combine results. These should be run after the main py files are run.
- The 'processed.csv' files in 'data/' are the same STATA dataframes, but are either altered to be .csv, have a column name change, or have character values converted to numeric.' 

**Note:** <br>
Most of the files cannot be run without a HPC. All scripts were designed to be run with 48 cores, 336 GB RAM, and 2 weeks of runtime. 
