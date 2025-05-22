
# 2. ### Model Estimation ###

### Packages ###
library(foreign)
library(readstata13)
library(sandwich)
library(lmtest)
library(rms)
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

# Table 1 (294) #
peha.boehmke <- robcov(lrm(adopt ~ srcs_decay + nbrs_lag + rpcpinc + totpop + legp_squire + citi6010 + unif_rep + unif_dem + time + time_sq + time_cube + state, x = TRUE, y = TRUE, data = boehmke.p2,tol=1e-20,maxit=1000), cluster = boehmke.p2$statepol); gc()


covariates <- c("srcs_decay","nbrs_lag","rpcpinc","totpop","legp_squire","citi6010","unif_rep","unif_dem","time","time_sq","time_cube")


boehmke_nona <- na.omit(boehmke.p2[,c(covariates,"adopt","statepol","policy","state")])

state_dummies <- dummy_cols(boehmke_nona$state)
state_dummies <- state_dummies[,-1]
names(state_dummies) <- substr(names(state_dummies),7,nchar(names(state_dummies)))

covariates <- c(covariates,names(state_dummies)[-1])

boehmke_nona<- data.frame(boehmke_nona,state_dummies)




# Table 1 (294) #
peha.boehmke <- robcov(lrm(adopt ~ srcs_decay + nbrs_lag + rpcpinc + totpop + legp_squire + citi6010 + unif_rep + unif_dem + time + time_sq + time_cube + state, x = TRUE, y = TRUE, data = boehmke_nona,tol=1e-200,maxit=1000), cluster = boehmke_nona$statepol); gc()

library(neha)

est0 <- glm(boehmke_nona$adopt ~ as.matrix(boehmke_nona[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

boehmke_nona <- neha_boehmke[[3]]

edge_effs <- neha_boehmke[[2]]

neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,edge_effs),collapse="+")))


neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,"edge_sum"),collapse="+")))

boehmke_nona$edge_sum <- apply(boehmke_nona[,edge_effs],1,sum)

neha.boehmke <- robcov(lrm(neha_formula, x = TRUE, y = TRUE, data = boehmke_nona), cluster = boehmke_nona$statepo)

peha_formula <- as.formula(paste("adopt ~",paste(c(covariates),collapse="+")))

peha.boehmke <- robcov(lrm(peha_formula, x = TRUE, y = TRUE, data = boehmke_nona), cluster = boehmke_nona$statepo)

minyr <- numeric(nrow(boehmke_nona))
for(i in 1:nrow(boehmke_nona)){
  minyr[i] <- min(boehmke_nona$year[which(boehmke_nona$policy==boehmke_nona$policy[i])])
}

boehmke_nona$minyr <- minyr

neha_formula_yrfun <- as.formula(paste("adopt ~",paste(c(covariates,"edge_sum","minyr"),collapse="+")))

neha.yrfun <- robcov(lrm(neha_formula_yrfun, x = TRUE, y = TRUE, data = boehmke_nona,tol=1e-20,maxit=1000), cluster = boehmke_nona$statepo)

save(list='neha.yrfun',file='./results_summaries/minyr_estimate.RData')



