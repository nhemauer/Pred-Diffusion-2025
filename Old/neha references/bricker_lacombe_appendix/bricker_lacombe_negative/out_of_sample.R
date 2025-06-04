################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 7/25/22                                                       #
################################################################################
### Packages ###
dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages("foreach", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("combinat", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("glmulti", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("doParallel", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("readstata13", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("MLmetrics", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )
install.packages("haven", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )

install.packages("fastglm", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" )

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'),repos = NULL, type='source')

library(foreach,lib.loc=Sys.getenv("R_LIBS_USER"))
library(doParallel,lib.loc=Sys.getenv("R_LIBS_USER"))
library(readstata13,lib.loc=Sys.getenv("R_LIBS_USER"))
library(MLmetrics,lib.loc=Sys.getenv("R_LIBS_USER"))
library(haven,lib.loc=Sys.getenv("R_LIBS_USER"))
library(neha,lib.loc=Sys.getenv("R_LIBS_USER"))


#library(readstata13)
#library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))
#library(foreach)
#library(doParallel)
#library(MLmetrics)
#library(haven)
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

covariates <- colnames(data_for_bl)[-ncol(data_for_bl)]

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


library(neha)

set.seed(100420)
upol <- unique(bl_nona$policyno)
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

#for(g in 1:max(pol_grp)){

for(g in 65:max(pol_grp)){

pol_g <- upol[which(pol_grp==g)]

pol_not_g <- upol[which(pol_grp!=g)]

data_train <- bl_nona[which(is.element(bl_nona$policyno,pol_not_g)),]


est0 <- glm(data_train$adoption ~ as.matrix(data_train[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

system.time(neha_bl_train <- neha(data_train,node="state",time="year",event="adoption",cascade="policyno",covariates=covariates,ncore=20,negative=T))

edges_inferred <- neha_bl_train[[2]]

print(edges_inferred)


save(list=c("neha_bl_train","data_train"),file=paste("./output_data/bl_oos_subset_results",g,".RData",sep=""))

     test_data <- data_neha_discrete(bl_nona[which(is.element(bl_nona$policyno,pol_g)),], node="state",time="year",event="adoption",cascade="policyno",covariates=covariates)

     a <- neha_bl_train[[1]]

     #edges_inferred <- edges_inferred[which(is.element(edges_inferred,names(test_data)))]

     neha_bl_train[[3]]$edge_sum_pos <- rep(0,nrow(neha_bl_train[[3]]))
     neha_bl_train[[3]]$edge_sum_neg <- rep(0,nrow(neha_bl_train[[3]]))
     
     covariates_neha <- c(covariates,edges_inferred)
     neha.train.s <- glm(neha_bl_train[[3]][,"adoption"] ~ as.matrix(neha_bl_train[[3]][, covariates_neha]),family="binomial")
     
     edge_sign <- sign(coef(neha.train.s)[-(1:(length(covariates)+1))])
     
     edges_inferred_pos <- edges_inferred[which(edge_sign>0)]
     edges_inferred_neg <- edges_inferred[which(edge_sign < 0)]
     
     test_data$edge_sum_pos <- rep(0,nrow(test_data))
     test_data$edge_sum_neg <- rep(0,nrow(test_data))

     for(i in 1:length(edges_inferred_pos)){
       neha_bl_train[[3]]$edge_sum_pos <- neha_bl_train[[3]]$edge_sum_pos + neha_bl_train[[3]][,edges_inferred_pos[i]]
       if(is.element(edges_inferred_pos[i],names(test_data))){
       test_data[,edges_inferred_pos[i]] <- (test_data[,edges_inferred_pos[i]]  > 0 )*exp(-exp(a)*test_data[,edges_inferred_pos[i]])
       test_data$edge_sum_pos <- test_data$edge_sum_pos + test_data[,edges_inferred_pos[i]]
       }
     }
     
     for(i in 1:length(edges_inferred_neg)){
       neha_bl_train[[3]]$edge_sum_neg <- neha_bl_train[[3]]$edge_sum_neg + neha_bl_train[[3]][,edges_inferred_neg[i]]
       if(is.element(edges_inferred_neg[i],names(test_data))){
         test_data[,edges_inferred_neg[i]] <- (test_data[,edges_inferred_neg[i]]  > 0 )*exp(-exp(a)*test_data[,edges_inferred_neg[i]])
         test_data$edge_sum_neg <- test_data$edge_sum_neg + test_data[,edges_inferred_neg[i]]
       }
     }

     covariates_neha <- c(covariates,"edge_sum_pos","edge_sum_neg")

     peha.train <-  glm(neha_bl_train[[3]][,"adoption"] ~ as.matrix(neha_bl_train[[3]][, covariates]),family="binomial")

     neha.train <- glm(neha_bl_train[[3]][,"adoption"] ~ as.matrix(neha_bl_train[[3]][, covariates_neha]),family="binomial")

     peha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates])) %*% cbind(coef(peha.train)) ))

     neha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha]) ) %*% cbind(coef(neha.train))  ))

     covariates_neha <- c(covariates,edges_inferred)

     neha.train.s <- glm(neha_bl_train[[3]][,"adoption"] ~ as.matrix(neha_bl_train[[3]][, covariates_neha]),family="binomial")

     covariates_neha_s <- covariates_neha[which(is.element(covariates_neha,names(test_data)))]

     coef_neha_s <- coef(neha.train.s)[c(1,which(is.element(covariates_neha,names(test_data)))+1)]

     neha.test.s <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha_s]) ) %*% cbind(coef_neha_s)))

     save(list=c("test_data","peha.test","neha.test","neha.test.s"),file=paste("./output_data/bl_subset_oos_predict",g,".RData",sep=""))

     y_test <- test_data[,"adoption"]

     testy <- c(testy,y_test)
     peha_scorey <- c(peha_scorey,peha.test)
     neha_scorey <- c(neha_scorey,neha.test)
     neha_scorey_s <- c(neha_scorey_s,neha.test.s)
     out_policy <- c(out_policy,rep(g,length(peha.test)))

     if(length(table(test_data[,"adoption"]))==2){
      peha_auc[g] <- PRAUC(peha.test,test_data[,"adoption"])
      neha_auc[g] <- PRAUC(neha.test,test_data[,"adoption"])
      neha_s_auc[g] <- PRAUC(neha.test.s,test_data[,"adoption"])
     }

     print(c(mean(peha_auc),mean(neha_auc),mean(neha_s_auc)))

print(g)

}

save(list=c("testy","neha_scorey","peha_scorey","neha_scorey_s"),file="./output_data/bl_oos_results.RData")
















