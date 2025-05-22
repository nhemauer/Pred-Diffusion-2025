################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 6/24/22                                                         #
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
karch <- read_dta("./replication_data/karch2016.dta")
karch <- karch[order(karch$state, karch$year), ]
karch[is.na(karch$stateyear), ]$stateyear <- 10

#compnum

options(na.action='na.pass')
data_for_karch <- model.matrix(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd,data=karch)[,-1]

colnames(data_for_karch) <- gsub(":","_",colnames(data_for_karch))

covariates <- colnames(data_for_karch)


data_for_karch <- data.frame(data_for_karch)
data_for_karch[,c("adopt","state","year","stateyear")] <- karch[,c("adopt","state","year","stateyear")]

data_for_karch$policy <- paste("p",karch$compnum,sep="")

karch_nona <- na.omit(data_for_karch)

system.time(neha_karch <- neha(karch_nona,node="state",time="year",event="adopt",cascade="policy",covariates=covariates,ncore=20))

save(list="neha_karch",file="./output_data/neha_karch_result.RData")
















