################################################################################
# 'A Diffusion Network Event History Estimator'                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analysis for Boushey (2016)                                      #
# Last update: 6/24/22                                                         #
################################################################################
### Packages ###

dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('readstata13', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('glmulti', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('combinat', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('glmnet', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('doParallel', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('foreach', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('boot', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('fastglm', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('MLmetrics', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('fastDummies', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )

install.packages('neha_0.1.0.tar.gz', repos = NULL, type='source')

library(readstata13)
library(neha)
library(foreach)
library(doParallel)
library(MLmetrics)
library(fastDummies)

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

## Boehmke et al. (2017) ##
boehmke.p2 <- read.dta13("./replication_data/boehmke2017.dta", convert.factors = FALSE)
boehmke.p2 <- boehmke.p2[order(boehmke.p2$state, boehmke.p2$year), ]
boehmke.p2$state <- factor(boehmke.p2$state)
boehmke.p2$state <- relevel(boehmke.p2$state, ref = "AL")


covariates <- c("srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire","citi6010","unif_rep","unif_dem","time","time_sq","time_cube")


boehmke_nona <- na.omit(boehmke.p2[,c(covariates,"adopt","statepol","policy","state","year")])

state_dummies <- dummy_cols(boehmke_nona$state)
state_dummies <- state_dummies[,-1]
names(state_dummies) <- substr(names(state_dummies),7,nchar(names(state_dummies)))

covariates <- c(covariates,names(state_dummies)[-1])

boehmke_nona<- data.frame(boehmke_nona,state_dummies)





library(neha)

est0 <- glm(boehmke_nona$adopt ~ as.matrix(boehmke_nona[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

system.time(neha_boehmke <- neha(boehmke_nona,node="state",time="year",event="adopt",cascade="policy",covariates=covariates,ncore=20))

save(list="neha_boehmke",file="./output_data/neha_boehmke_result.RData")





