The files in this directory can be used to reproduce the results of the Bricker/LaCombe replication. The directory structure is organized as follows. 

1. main directory: contains other directories as well as all of the code required to reproduce the Boushey replication and simulation results. 

2. output_data: contains intermediate files written by individual R scripts.

3. replication_data: contains original replication data from Bricker/LaCombe.

4. results_summaries: contains final analysis summary files from which the statistics, tables, and figures in the paper are directly generated.

It is assumed that each R script is run with the working directory set to the top-level directory. The pbs scripts are organized to run the analyses parallelized across twenty cores. The R scripts can be run independently, without calling the pbs scripts. Note that R package installation lines in the R scripts may need to be modified based on the system on which R is being run. 

There are four core analysis scripts that are time-consuming to run (2-100hrs when spread across 20 cores), but can be run simultaneously. 

1. replication.R: just runs neha on the replication data. This should run in less than 2hrs.

2. out_of_sample.R: runs the leave-one-policy-out prediction experiments. Takes 24-48hrs based on the system.

3. simulation.R: runs the part of the simulation study in which the diffusion ties are active. Takes around 80-100 hours based on the system.

4. simulation0.R: runs the part of the simulation study in which the diffusion ties are not active. Takes around 80-100 hours based on the system.

There are two post-analysis scripts that should be run following scripts 2--4. Script 1 is run by Script 3, so needn't be run on its own. The following two scripts were not run on HPC, and so there are no pbs scripts.

1. interpretation.R: produces all of the results summaries that appear in the main component of the paper.

2. policy_prediction_comparison.R: produces the extended analysis of cascade timing that appears in the appendix.

There are two Stata .do files that need to be run to produce the random effect logit model results. 

1. bl_melogit_replication.do produces the regression table.

2. simulate-brickerlacombe01.do produces the interpretation of the signatures variable in Figure 2 of the paper. Must be run after bl_melogit_replication.do, and takes around five hours to run.

