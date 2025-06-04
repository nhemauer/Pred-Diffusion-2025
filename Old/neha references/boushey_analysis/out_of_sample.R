################################################################################
# 'A Diffusion Network Event History Estimator'                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Predictionn analysis for Boushey (2016)                                      #
# Last update: 6/22/22                                                         #
################################################################################
### Packages ###

dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('neha_0.1.0.tar.gz',Sys.getenv('R_LIBS_USER'), repos = NULL, type='source')

library(readstata13)
library(foreach)
library(doParallel)
library(MLmetrics)
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))


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


## Boushey (2016) ##
boushey <- read.dta13("replication_data/boushey2016.dta")
boushey <- boushey[order(boushey$state, boushey$year), ]


covariates <- c("policycongruent","gub_election","elect2","fedcrime","leg_dem_per_2pty","dem_governor","insession","propneighpol","citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq","violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3")


boushey_nona <- na.omit(boushey[,c(covariates,"state","year","dvadopt","billname")])

boushey_nona$state <- gsub(" ", ".", boushey_nona$state)

boushey_nona_all <- boushey_nona

set.seed(100420)
upol <- unique(boushey_nona_all$billname)
#pol_grp <- sample(1:5,length(upol),rep=T)
pol_grp <- 1:length(upol)

peha_auc <- numeric(length(upol))
neha_auc <- numeric(length(upol))
neha_s_auc <- numeric(length(upol))

testy <- NULL
peha_scorey <- NULL
neha_scorey <- NULL
neha_scorey_s <- NULL

cvll.diff <- NULL

out_policy <- NULL

for(g in 1:length(unique(pol_grp))){

pol_g <- upol[which(pol_grp==g)]

pol_not_g <- upol[which(pol_grp!=g)]

data_train <- boushey_nona[which(is.element(boushey_nona$billname,pol_not_g)),]

system.time(neha_boushey_train <- neha(data_train,node="state",time="year",event="dvadopt",cascade="billname",covariates=covariates,ncore=18))

edges_inferred <- neha_boushey_train[[2]]

print(edges_inferred)


save(list=c("neha_boushey_train","data_train"),file=paste("./output_data/boushey_oos_subset_results",g,".RData",sep=""))

     test_data <- data_neha_discrete(boushey_nona[which(is.element(boushey_nona$billname,pol_g)),], node="state",time="year",event="dvadopt",cascade="billname",covariates=covariates)

     a <- neha_boushey_train[[1]]

     print(a)

     covariates_neha <- c(covariates,edges_inferred)

     edge_sum <- rep(0,nrow(test_data))
     for(i in 1:length(edges_inferred)){
       if(is.element(edges_inferred[i],names(test_data))){
       test_data[,edges_inferred[i]] <- (test_data[,edges_inferred[i]]  > 0 )*exp(-exp(a)*test_data[,edges_inferred[i]])
       edge_sum <- edge_sum +test_data[,edges_inferred[i]]
       }
     }
     test_data$edge_sum <- edge_sum

     edge_sum <- rep(0,nrow(neha_boushey_train[[3]]))
     for(i in 1:length(edges_inferred)){
       edge_sum <- edge_sum + neha_boushey_train[[3]][,edges_inferred[i]]
     }
     neha_boushey_train[[3]]$edge_sum <- edge_sum

     covariates_neha <- c(covariates,"edge_sum")

     peha.train <-  glm(neha_boushey_train[[3]][,"dvadopt"] ~ as.matrix(neha_boushey_train[[3]][, covariates]),family="binomial")

     neha.train <- glm(neha_boushey_train[[3]][,"dvadopt"] ~ as.matrix(neha_boushey_train[[3]][, covariates_neha]),family="binomial")

     covariates_neha_s <- c(covariates,edges_inferred)

     neha.train.s <- glm(neha_boushey_train[[3]][,"dvadopt"] ~ as.matrix(neha_boushey_train[[3]][, covariates_neha_s]),family="binomial")

     t_covariates_neha_s <- covariates_neha_s[which(is.element(covariates_neha_s,names(test_data)))]
     t_coef_neha_s <- coef(neha.train.s)[c(1,which(is.element(covariates_neha_s,names(test_data)))+1)]

     peha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates])) %*% cbind(coef(peha.train)) ))

     neha.test <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, covariates_neha]) ) %*% cbind(coef(neha.train))  ))

     neha.test.s <- 1/(1+exp(-  cbind(1, as.matrix(test_data[, t_covariates_neha_s]) ) %*% cbind(t_coef_neha_s)  ))

     save(list=c("test_data","peha.test","neha.test","neha.test.s"),file=paste("./output_data/boushey_subset_oos_predict75",g,".RData",sep=""))

     y_test <- test_data[,"dvadopt"]

     testy <- c(testy,y_test)
     peha_scorey <- c(peha_scorey,peha.test)
     neha_scorey <- c(neha_scorey,neha.test)
     neha_scorey_s <- c(neha_scorey_s,neha.test.s)
     out_policy <- c(out_policy,rep(g,length(peha.test)))

     peha_auc[g] <- PRAUC(peha.test,test_data[,"dvadopt"])
     neha_auc[g] <- PRAUC(neha.test,test_data[,"dvadopt"])
     neha_s_auc[g] <- PRAUC(neha.test.s,test_data[,"dvadopt"])

      print(c(mean(peha_auc),mean(neha_auc),mean(neha_s_auc)))

print(g)

}

save(list=c("testy","neha_scorey","peha_scorey","neha_scorey_s","out_policy"),file="./output_data/boushey_subset_oos_results_combined.RData")

