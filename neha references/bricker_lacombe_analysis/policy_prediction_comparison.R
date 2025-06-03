# 2. ### Model Estimation ###

load('./output_data/neha_bl_result.RData')

### Packages ###
library(foreign)
library(readstata13)
library(sandwich)
library(haven)
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


bl <- read_dta("./replication_data/monadic_analysis_largen.dta")



# Table 2, column 1 (90) #
options(na.action='na.pass')
data_for_bl <- model.matrix(adoption ~ std_score + std_score + initiative + init_sigs + std_population + std_citideology + unified + std_income + std_legp_squire + duration+ durationsq + durationcb + as.factor(year),data=bl)[,-1]

colnames(data_for_bl) <- gsub("as.factor\\(year\\)","y",colnames(data_for_bl))

covariates <- colnames(data_for_bl)[-ncol(data_for_bl)]

data_for_bl <- data.frame(data_for_bl)
data_for_bl[,c("adoption","state","year","policyno")] <- bl[,c("adoption","state","year","policyno")]

data_for_bl$policyno <- paste("p",data_for_bl$policyno,sep="")

bl_nona <- na.omit(data_for_bl)

library(neha)

bl_nona <- neha_bl[[3]]

edge_effs <- neha_bl[[2]]

neha_formula <- as.formula(paste("adoption ~",paste(c(covariates,"edge_sum"),collapse="+")))

bl_nona$edge_sum <- apply(bl_nona[,edge_effs],1,sum)

neha.bl<- glm(neha_formula, x = TRUE, y = TRUE, data = bl_nona,family="binomial")

minyr <- numeric(nrow(bl_nona))

for(i in 1:length(minyr)){
  minyr[i] <- min(bl_nona$year[which(bl_nona$policyno==bl_nona$policyno[i])])
}

bl_nona$minyr <- minyr

neha_formula_yrfun <- as.formula(paste("adoption ~",paste(c(covariates,"edge_sum","minyr"),collapse="+")))

neha.bl.yrfun <- glm(neha_formula_yrfun, x = TRUE, y = TRUE, data = bl_nona,family="binomial")
