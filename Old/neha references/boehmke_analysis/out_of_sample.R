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
install.packages('MLmetrics', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('stringi', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
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

covariates_all <- c(covariates,names(state_dummies)[-1])
covariates <- covariates_all

boehmke_nona<- data.frame(boehmke_nona,state_dummies)


library(neha)

set.seed(100420)
upol <- unique(boehmke_nona$policy)
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

data_train <- boehmke_nona[which(is.element(boehmke_nona$policy,pol_not_g)),]

covariates <- covariates_all

est0 <- glm(data_train$adopt ~ as.matrix(data_train[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

system.time(neha_boehmke_train <- neha(data_train,node="state",time="year",event="adopt",cascade="policy",covariates=covariates,ncore=20))

edges_inferred <- neha_boehmke_train[[2]]

print(edges_inferred)


save(list=c("neha_boehmke_train","data_train"),file=paste("./output_data/boehmke_oos_subset_results",g,".RData",sep=""))

     test_data <- data_neha_discrete(boehmke_nona[which(is.element(boehmke_nona$policy,pol_g)),], node="state",time="year",event="adopt",cascade="policy",covariates=covariates)

     a <- neha_boehmke_train[[1]]

     #edges_inferred <- edges_inferred[which(is.element(edges_inferred,names(test_data)))]

     neha_boehmke_train[[3]]$edge_sum <- rep(0,nrow(neha_boehmke_train[[3]]))
     test_data$edge_sum <- rep(0,nrow(test_data))

     for(i in 1:length(edges_inferred)){
       neha_boehmke_train[[3]]$edge_sum <- neha_boehmke_train[[3]]$edge_sum + neha_boehmke_train[[3]][,edges_inferred[i]]
       if(is.element(edges_inferred[i],names(test_data))){
       test_data[,edges_inferred[i]] <- (test_data[,edges_inferred[i]]  > 0 )*exp(-exp(a)*test_data[,edges_inferred[i]])
       test_data$edge_sum <- test_data$edge_sum + test_data[,edges_inferred[i]]
       }
     }

     covariates_neha <- c(covariates,"edge_sum")

     peha.train <-  glm(neha_boehmke_train[[3]][,"adopt"] ~ as.matrix(neha_boehmke_train[[3]][, covariates]),family="binomial")

     neha.train <- glm(neha_boehmke_train[[3]][,"adopt"] ~ as.matrix(neha_boehmke_train[[3]][, covariates_neha]),family="binomial")

     peha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates])) %*% cbind(coef(peha.train)) ))

     neha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha]) ) %*% cbind(coef(neha.train))  ))

     covariates_neha <- c(covariates,edges_inferred)

     neha.train.s <- glm(neha_boehmke_train[[3]][,"adopt"] ~ as.matrix(neha_boehmke_train[[3]][, covariates_neha]),family="binomial")


     covariates_neha_s <- covariates_neha[which(is.element(covariates_neha,names(test_data)))]
     coef_s <- coef(neha.train.s)[c(1,which(is.element(covariates_neha,names(test_data))) + 1)]

     neha.test.s <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha_s]) ) %*% cbind(coef_s)  ))

     save(list=c("test_data","peha.test","neha.test","neha.test.s"),file=paste("./output_data/boehmke_subset_oos_predict",g,".RData",sep=""))

     y_test <- test_data[,"adopt"]

     testy <- c(testy,y_test)
     peha_scorey <- c(peha_scorey,peha.test)
     neha_scorey <- c(neha_scorey,neha.test)
     neha_scorey_s <- c(neha_scorey_s,neha.test.s)
     out_policy <- c(out_policy,rep(g,length(peha.test)))

     peha_auc[g] <- PRAUC(peha.test,test_data[,"adopt"])
     neha_auc[g] <- PRAUC(neha.test,test_data[,"adopt"])
     neha_s_auc[g] <- PRAUC(neha.test.s,test_data[,"adopt"])

     print(c(mean(peha_auc),mean(neha_auc),mean(neha_s_auc)))




print(g)

}

save(list=c("testy","neha_scorey","peha_scorey","neha_scorey_s"),file="./output_data/boehmke_oos_results.RData")

