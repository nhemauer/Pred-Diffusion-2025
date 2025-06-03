The files in this directory can be used to reproduce the results of the Bricker/LaCombe analyses that appear in the online appendix. The directory structure is organized as follows. 

1. main directory: contains other directories as well as all of the code required to reproduce the results. 

2. bricker_lacombe_negative: contains files required to reproduce the analysis that infers negative diffusion ties.

It is assumed that each R script is run with the working directory set to directory in which the R script is stored. The two R scripts in the current folder must be run after all of the R scripts in the bricker_lacombe_analysis folder have finished running. There are two R scripts in this folder. They can be run simultaneously and each takes less than 30min to complete.

1. bricker_lacombe_netanalysis.R: runs the analysis of the association between legislative professionalism and diffusion network tie structure.

2. bricker_lacombe_omitted_variable_analysis.R: runs the analysis of the relative performance of peha and neha in terms of covariate coefficient estimation performance.


