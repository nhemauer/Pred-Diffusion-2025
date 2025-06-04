The files in this directory can be used to reproduce the results of the Boehmke (2017) replication. The directory structure is organized as follows. 

1. main directory: contains other directories as well as all of the code required to reproduce the Boehmke replication and simulation results. 

2. output_data: contains intermediate files written by individual R scripts.

3. replication_data: contains original replication data from Boehmke (201y).

4. results_summaries: contains final analysis summary files from which the statistics, tables, and figures in the paper are directly generated.

It is assumed that each R script is run with the working directory set to the top-level directory. The pbs scripts are organized to run the analyses parallelized across twenty cores. The R scripts can be run independently, without calling the pbs scripts. Note that R package installation lines in the R scripts may need to be modified based on the system on which R is being run. 

There are five core analysis scripts that are time-consuming to run (1-24hrs when spread across 20 cores), but can be run simultaneously. 

1. replication.R: just runs neha on the Boushey (2016) replication data. This should run in less than 1hr.

2. out_of_sample.R: runs the leave-one-policy-out prediction experiments. Takes 12-24hrs based on the system.

3. simulation.R: runs the part of the simulation study in which the diffusion ties are active. Takes around 12-24 hours based on the system.

3. simulation0.R: runs the part of the simulation study in which the diffusion ties are not active. Takes around 12-24 hours based on the system.

There are two post-analysis scripts that should be run following scripts 2--4. Script 1 is run by Script 3, so needn't be run on its own. The following two scripts were not run on HPC, and so there are no pbs scripts.

1. interpret.R: produces all of the results summaries that appear in the main component of the paper.

2. policy_prediction_comparison.R: produces the extended analysis of cascade timing that appears in the appendix.