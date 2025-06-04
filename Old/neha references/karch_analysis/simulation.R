################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred karch, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 6/24/22                                                        #
################################################################################
### Packages ###
dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'),repos = NULL, type='source')

source('replication.R')


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

library(neha)

load("./output_data/neha_karch_result.RData")

neha_formula <- as.formula(paste("adopt~",paste(c(covariates,"edge_sum"),collapse="+"),sep=""))

edge_effs  <- neha_karch[[2]]

neha_karch[[3]]$edge_sum <- apply(neha_karch[[3]][,edge_effs],1,sum)

neha_coef <- coef(glm(neha_formula,family=binomial,data=neha_karch[[3]]))

# simulate a cascade
ucascades <- unique(karch_nona$policy)
karch_nona$Intercept <- 1

beta <- cbind(neha_coef[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(neha_coef[-(1:(length(covariates)+1))],length(edge_effs)))
rownames(gamma) <- edge_effs

a_par <- neha_karch[[1]]

# simulate data

nsim <- 50
set.seed(9202016)

precis <- NULL
recall <- NULL

for(s in 1:nsim){

  sim_karch <- NULL

  simulated_cascades <- list()
  for(c  in 1:length(ucascades)){

    karch_nona_c <- karch_nona[which(karch_nona$policy==ucascades[c]),]




    times <- min(karch_nona_c$year):max(karch_nona_c$year)
    times <- min(karch_nona_c$year):(max(karch_nona_c$year)+length(times))
    nodes <- unique(karch_nona_c$state)

    sim_base_c <- NULL
    sim_time <- NULL
    for(n in nodes){
      for(t in times){
        row_n <- which(karch_nona_c$state==n)
        row_nt <- sample(rep(row_n,2),1)
        sim_base_c <- rbind(sim_base_c,karch_nona_c[row_nt,])
        sim_time <- c(sim_time,t)
      }
    }

    sim_base_c$sim_time <- sim_time-min(times) +1

    simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

    simulated_cascades[[c]] <- simulated_cascade

  }

  sim_karch <- do.call('rbind',simulated_cascades)

  print("finished cascade simulation")

  neha_karch_sim <- neha(sim_karch,node="state",time="sim_time",event="event",cascade="policy",covariates=covariates,ncore=20)

  edges_subset <- neha_karch_sim[[2]]

  precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

  print(precis)

  recall <- c(recall,mean(is.element(edge_effs,edges_subset)))

  print(recall)

  save(list=c("edges_subset","neha_karch_sim"),file=paste("./output_data/neha_karch_sim_subset",s,".RData",sep=""))

  print(s)

}






##### ZERO #####
gamma <- cbind(rep(0,length(edge_effs)))
rownames(gamma) <- edge_effs

a_par <- neha_karch[[1]]

# simulate data

nsim <- 50
set.seed(9202016)

precis <- NULL
recall <- NULL

for(s in 1:nsim){

  sim_karch <- NULL

  simulated_cascades <- list()
  for(c  in 1:length(ucascades)){

    karch_nona_c <- karch_nona[which(karch_nona$policy==ucascades[c]),]




    times <- min(karch_nona_c$year):max(karch_nona_c$year)
    times <- min(karch_nona_c$year):(max(karch_nona_c$year)+length(times))
    nodes <- unique(karch_nona_c$state)

    sim_base_c <- NULL
    sim_time <- NULL
    for(n in nodes){
      for(t in times){
        row_n <- which(karch_nona_c$state==n)
        row_nt <- sample(rep(row_n,2),1)
        sim_base_c <- rbind(sim_base_c,karch_nona_c[row_nt,])
        sim_time <- c(sim_time,t)
      }
    }

    sim_base_c$sim_time <- sim_time-min(times) +1

    simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

    simulated_cascades[[c]] <- simulated_cascade

  }

  sim_karch <- do.call('rbind',simulated_cascades)

  print("finished cascade simulation")

  neha_karch_sim <- neha(sim_karch,node="state",time="sim_time",event="event",cascade="policy",covariates=covariates,ncore=4)

  edges_subset <- neha_karch_sim[[2]]

  precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

  print(precis)

  recall <- c(recall,mean(is.element(edge_effs,edges_subset)))

  print(recall)

  save(list=c("edges_subset","neha_karch_sim"),file=paste("./output_data/neha_karch_sim_subset_zero",s,".RData",sep=""))

  print(s)

}
















