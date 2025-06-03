################################################################################
# 'A Diffusion Network Event History Estimator'                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analysis for Boushey (2016)                                      #
# Last update: 6/22/22                                                         # 
################################################################################
### Packages ###

dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('readstata13', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('glmulti', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('combinat', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('glmnet', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('doParallel', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('foreach', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('boot', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('fastglm', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'), repos = NULL, type='source')

library(readstata13,lib.loc=Sys.getenv('R_LIBS_USER'))
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))
library(foreach,lib.loc=Sys.getenv('R_LIBS_USER'))
library(doParallel,lib.loc=Sys.getenv('R_LIBS_USER'))

# setwd()
# options(scipen = 99)

## Functions ##
substrRight <- function(x, n){
  substr(x, nchar(x) - n + 1, nchar(x))
}

## Boushey (2016) ##
boushey <- read.dta13('replication_data/boushey2016.dta')
boushey <- boushey[order(boushey$state, boushey$year), ]

covariates <- c('policycongruent','gub_election','elect2','fedcrime','leg_dem_per_2pty','dem_governor','insession','propneighpol','citidist','squire_prof86','citi6008','crimespendpc','crimespendpcsq','violentthousand','pctwhite','stateincpercap','logpop','counter','counter2','counter3')

boushey_nona <- na.omit(boushey[,c(covariates,'state','year','dvadopt','billname','styear')])

set.seed(55553322)

boushey_nona$state <- gsub(' ','.',boushey_nona$state)

system.time(neha_boushey <- neha(boushey_nona,node='state',time='year',event='dvadopt',cascade='billname',covariates=covariates,ncore=20))

save(list='neha_boushey',file='./output_data/neha_result.RData')





