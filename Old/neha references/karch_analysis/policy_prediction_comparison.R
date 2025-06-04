
library(foreign)
library(readstata13)
library(sandwich)
library(lmtest)
library(rms)
library(neha)

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
karch <- read.dta("./replication_data/karch2016.dta")
karch <- karch[order(karch$state, karch$year), ] 
karch[is.na(karch$stateyear), ]$stateyear <- 10

#compnum

# Table 2, column 1 (90) #
peha.karch <- robcov(lrm(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd, x = TRUE, y = TRUE, data = karch), cluster = karch$stateyear)

options(na.action='na.pass')
data_for_karch <- model.matrix(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd,data=karch)[,-1]

colnames(data_for_karch) <- gsub(":","_",colnames(data_for_karch))

covariates <- colnames(data_for_karch)


data_for_karch <- data.frame(data_for_karch)
data_for_karch[,c("adopt","state","year","stateyear")] <- karch[,c("adopt","state","year","stateyear")]

data_for_karch$policy <- paste("p",karch$compnum,sep="")

karch_nona <- na.omit(data_for_karch)


library(neha)

karch_nona <- neha_karch[[3]]

edge_effs <- neha_karch[[2]]

neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,edge_effs),collapse="+")))


neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,"edge_sum"),collapse="+")))

karch_nona$edge_sum <- apply(karch_nona[,edge_effs],1,sum)

neha.karch <- robcov(lrm(neha_formula, x = TRUE, y = TRUE, data = karch_nona), cluster = karch_nona$stateyear)

minyr <- numeric(nrow(karch_nona))
for(i in 1:length(minyr)){
  minyr[i] <- min(karch_nona$year[which(karch_nona$policy==karch_nona$policy[i])])
}

karch_nona$minyr <- minyr

neha_formula_yrfun <- as.formula(paste("adopt ~",paste(c(covariates,"edge_sum","minyr"),collapse="+")))

neha.yrfun <- robcov(lrm(neha_formula_yrfun, x = TRUE, y = TRUE, data = karch_nona,tol=1e-200,maxit=10000), cluster = karch_nona$stateyear)

save(list='neha.yrfun',file='./results_summaries/minyr_estimate.RData')

