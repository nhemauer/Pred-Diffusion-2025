################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 6/23/22                                                         #
################################################################################
### Packages ###
#install.packages('neha_0.1.0.tar.gz', repos = NULL, type='source')
dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'),repos = NULL, type='source')



library(readstata13)
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))
library(haven)
library(foreach)
library(doParallel)
library(MLmetrics)

# setwd()
options(scipen = 99)

## Functions ##
substrRight <- function(x, n){
  substr(x, nchar(x) - n + 1, nchar(x))
}

rcse <- function(model, cluster){
  require(sandwich); require(lmtest)
  G <- length(unique(cluster))
  N <- length(resid(model))
  k <- length(coef(model))
  dfa <- (G/(G - 1)) * (N - 1)/model$df.residual
  u <- estfun(model)
  u.clust <- matrix(NA, nrow = G, ncol = k)
  for(j in 1:k){
    u.clust[ , j] <- tapply(u[ , j], cluster, sum)
  }
  rcov <- dfa * vcov(model) %*% t(u.clust) %*% (u.clust) %*% vcov(model)
  rcse <- sqrt(diag(rcov))
  result <- coeftest(model, vcov = rcov)
  return(list(rcov = rcov, rcse = rcse, result = result))
}


## Karch et al. (2016) ##
bl <- read_dta("./replication_data/monadic_analysis_largen.dta")

# melogit adoption std_score initiative init_sigs std_pop std_citideology unified std_income std_legp_squire duration  durationsq durationcb i.year || policyno:

options(na.action='na.pass')
data_for_bl <- model.matrix(adoption ~ std_score + std_score + initiative + init_sigs + std_population + std_citideology + unified + std_income + std_legp_squire + duration+ durationsq + durationcb + as.factor(year),data=bl)[,-1]

colnames(data_for_bl) <- gsub("as.factor\\(year\\)","y",colnames(data_for_bl))

covariates <- colnames(data_for_bl)

data_for_bl <- data.frame(data_for_bl)
data_for_bl[,c("adoption","state","year","policyno")] <- bl[,c("adoption","state","year","policyno")]

data_for_bl$policyno <- paste("p",data_for_bl$policyno,sep="")

bl_nona <- na.omit(data_for_bl)

## WORKAROUND: https://github.com/rstudio/rstudio/issues/6692
## Revert to 'sequential' setup of PSOCK cluster in RStudio Console on macOS and R 4.0.0
if (Sys.getenv("RSTUDIO") == "1" && !nzchar(Sys.getenv("RSTUDIO_TERM")) && 
    Sys.info()["sysname"] == "Darwin" && getRversion() >= "4.0.0") {
  parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
}

system.time(neha_bl <- neha(bl_nona,node="state",time="year",event="adoption",cascade="policyno",covariates=covariates[-length(covariates)],ncore=6,negative=T))

save(list="neha_bl",file="./output_data/neha_bl_result.RData")








