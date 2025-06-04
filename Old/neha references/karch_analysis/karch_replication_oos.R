################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 3/27/19                                                         #
################################################################################
### Packages ###
dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'),repos = NULL, type='source')

library(readstata13)
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))
library(foreach)
library(doParallel)
library(MLmetrics)
library(haven)
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
karch <- read_dta("karch2016.dta")
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

library(neha)

set.seed(100420)
upol <- unique(karch_nona$policy)
pol_grp <- 1:length(upol)

peha_auc <- numeric(length(upol))
neha_auc <- numeric(length(upol))
neha_s_auc <- numeric(length(upol))

testy <- NULL
peha_scorey <- NULL
neha_scorey <- NULL
neha_scorey_s <- NULL
out_policy <- NULL

cvll.diff <- NULL

for(g in 1:max(pol_grp)){

pol_g <- upol[which(pol_grp==g)]

pol_not_g <- upol[which(pol_grp!=g)]

data_train <- karch_nona[which(is.element(karch_nona$policy,pol_not_g)),]


est0 <- glm(data_train$adopt ~ as.matrix(data_train[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

system.time(neha_karch_train <- neha(data_train,node="state",time="year",event="adopt",cascade="policy",covariates=covariates,ncore=20))

edges_inferred <- neha_karch_train[[2]]

print(edges_inferred)


save(list=c("neha_karch_train","data_train"),file=paste("karch_oos_subset_results",g,".RData",sep=""))

     test_data <- data_neha_discrete(karch_nona[which(is.element(karch_nona$policy,pol_g)),], node="state",time="year",event="adopt",cascade="policy",covariates=covariates)

     a <- neha_karch_train[[1]]

     #edges_inferred <- edges_inferred[which(is.element(edges_inferred,names(test_data)))]

     neha_karch_train[[3]]$edge_sum <- rep(0,nrow(neha_karch_train[[3]]))
     test_data$edge_sum <- rep(0,nrow(test_data))

     for(i in 1:length(edges_inferred)){
       neha_karch_train[[3]]$edge_sum <- neha_karch_train[[3]]$edge_sum + neha_karch_train[[3]][,edges_inferred[i]]
       if(is.element(edges_inferred[i],names(test_data))){
       test_data[,edges_inferred[i]] <- (test_data[,edges_inferred[i]]  > 0 )*exp(-exp(a)*test_data[,edges_inferred[i]])
       test_data$edge_sum <- test_data$edge_sum + test_data[,edges_inferred[i]]
       }
     }

     covariates_neha <- c(covariates,"edge_sum")

     peha.train <-  glm(neha_karch_train[[3]][,"adopt"] ~ as.matrix(neha_karch_train[[3]][, covariates]),family="binomial")

     neha.train <- glm(neha_karch_train[[3]][,"adopt"] ~ as.matrix(neha_karch_train[[3]][, covariates_neha]),family="binomial")

     peha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates])) %*% cbind(coef(peha.train)) ))

     neha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha]) ) %*% cbind(coef(neha.train))  ))

     covariates_neha <- c(covariates,edges_inferred)

     neha.train.s <- glm(neha_karch_train[[3]][,"adopt"] ~ as.matrix(neha_karch_train[[3]][, covariates_neha]),family="binomial")

     covariates_neha_s <- covariates_neha[which(is.element(covariates_neha,names(test_data)))]

     coef_neha_s <- coef(neha.train.s)[c(1,which(is.element(covariates_neha,names(test_data)))+1)]

     neha.test.s <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha_s]) ) %*% cbind(coef_neha_s)))

     save(list=c("test_data","peha.test","neha.test","neha.test.s"),file=paste("karch_subset_oos_predict",g,".RData",sep=""))

     y_test <- test_data[,"adopt"]

     testy <- c(testy,y_test)
     peha_scorey <- c(peha_scorey,peha.test)
     neha_scorey <- c(neha_scorey,neha.test)
     neha_scorey_s <- c(neha_scorey_s,neha.test.s)
     out_policy <- c(out_policy,rep(g,length(peha.test)))

     if(length(table(test_data[,"adopt"]))==2){
      peha_auc[g] <- PRAUC(peha.test,test_data[,"adopt"])
      neha_auc[g] <- PRAUC(neha.test,test_data[,"adopt"])
      neha_s_auc[g] <- PRAUC(neha.test.s,test_data[,"adopt"])
     }

     print(c(mean(peha_auc),mean(neha_auc),mean(neha_s_auc)))

print(g)

}

save(list=c("testy","neha_scorey","peha_scorey","neha_scorey_s"),file="karch_oos_results.RData")
















