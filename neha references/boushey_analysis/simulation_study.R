################################################################################
# 'A Diffusion Network Event History Estimator'                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Simulation study for Boushey (2016)                                          #
# Last update: 6/22/22                                                         #
################################################################################

# run replicaiton
source('replication.R')

### Packages ###

dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('neha_0.1.0.tar.gz',Sys.getenv('R_LIBS_USER'), repos = NULL, type='source')

library(readstata13)
library(foreach)
library(doParallel)
library(MLmetrics)
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))


# setwd()
# options(scipen = 99)

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
boushey <- read.dta13("./replication_data/boushey2016.dta")
boushey <- boushey[order(boushey$state, boushey$year), ]

covariates <- c("policycongruent","gub_election","elect2","fedcrime","leg_dem_per_2pty","dem_governor","insession","propneighpol","citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq","violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3")

boushey_nona <- na.omit(boushey[,c(covariates,"state","year","dvadopt","billname","styear")])


set.seed(55553322)

boushey_nona$state <- gsub(" ",".",boushey_nona$state)

load("./output_data/neha_result.RData")

boushey_nona <- neha_boushey[[3]]

edge_effs <- neha_boushey[[2]]

boushey_nona$edge_sum <- apply(boushey_nona[,edge_effs],1,sum)

neha_formula <- as.formula(paste("dvadopt ~",paste(c(covariates,"edge_sum"),collapse="+")))

neha.boushey <- glm(neha_formula, x = TRUE, y = TRUE, data = boushey_nona,family=binomial)

boushey_nona <- na.omit(boushey[,c(covariates,"state","year","dvadopt","billname","styear")])

set.seed(55553322)

boushey_nona$state <- gsub(" ",".",boushey_nona$state)


# simulate a cascade
ucascades <- unique(boushey_nona$billname)
boushey_nona$Intercept <- 1

beta <- cbind(coef(neha.boushey)[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(coef(neha.boushey)[-(1:(length(covariates)+1))],length(edge_effs)))

rownames(gamma) <- edge_effs

a_par <- neha_boushey[[1]]

# simulate data

nsim <- 50
set.seed(10042016)

precis <- NULL
recall <- NULL

for(s in 1:nsim){

sim_boushey <-NULL

simulated_cascades <- list()
for(c  in 1:length(ucascades)){

boushey_nona_c <- boushey_nona[which(boushey_nona$billname==ucascades[c]),]




times <- min(boushey_nona_c$year):max(boushey_nona_c$year)
times <- min(boushey_nona_c$year):(max(boushey_nona_c$year)+length(times))
nodes <- unique(boushey_nona_c$state)

sim_base_c <- NULL
sim_time <- NULL
for(n in nodes){
  for(t in times){
    row_n <- which(boushey_nona_c$state==n)
    row_nt <- sample(rep(row_n,2),1)
    sim_base_c <- rbind(sim_base_c,boushey_nona_c[row_nt,])
    sim_time <- c(sim_time,t)
  }
}

sim_base_c$sim_time <- sim_time-min(times) +1

simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

simulated_cascades[[c]] <- simulated_cascade

}

sim_boushey <- do.call('rbind',simulated_cascades)

print("finished cascade simulation")

neha_boushey_sim <- neha(sim_boushey,node="state",time="sim_time",event="event",cascade="billname",covariates=covariates,ncore=20)

edges_subset <- neha_boushey_sim[[2]]

precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

print(precis)

recall <- c(recall,mean(is.element(edge_effs,edges_subset)))

print(recall)

save(list=c("edges_subset","neha_boushey_sim"),file=paste("./output_data/neha_boushey_sim_subset",s,".RData",sep=""))

print(s)

}


#### Zero Sim ####

## Boushey (2016) ##
boushey <- read.dta13("./replication_data/boushey2016.dta")
boushey <- boushey[order(boushey$state, boushey$year), ]

covariates <- c("policycongruent","gub_election","elect2","fedcrime","leg_dem_per_2pty","dem_governor","insession","propneighpol","citidist","squire_prof86","citi6008","crimespendpc","crimespendpcsq","violentthousand","pctwhite","stateincpercap","logpop","counter","counter2","counter3")

boushey_nona <- na.omit(boushey[,c(covariates,"state","year","dvadopt","billname","styear")])


set.seed(55553322)

boushey_nona$state <- gsub(" ",".",boushey_nona$state)

load("./output_data/neha_result.RData")

boushey_nona <- neha_boushey[[3]]

edge_effs <- neha_boushey[[2]]

boushey_nona$edge_sum <- apply(boushey_nona[,edge_effs],1,sum)

neha_formula <- as.formula(paste("dvadopt ~",paste(c(covariates,"edge_sum"),collapse="+")))

neha.boushey <- glm(neha_formula, x = TRUE, y = TRUE, data = boushey_nona,family=binomial)

boushey_nona <- na.omit(boushey[,c(covariates,"state","year","dvadopt","billname","styear")])

set.seed(55553322)

boushey_nona$state <- gsub(" ",".",boushey_nona$state)


# simulate a cascade
ucascades <- unique(boushey_nona$billname)
boushey_nona$Intercept <- 1

beta <- cbind(coef(neha.boushey)[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(0,length(edge_effs)))

rownames(gamma) <- edge_effs

a_par <- neha_boushey[[1]]

# simulate data

nsim <- 50
set.seed(10042016)

precis <- NULL
recall <- NULL

for(s in 1:nsim){

sim_boushey <-NULL

simulated_cascades <- list()
for(c  in 1:length(ucascades)){

boushey_nona_c <- boushey_nona[which(boushey_nona$billname==ucascades[c]),]




times <- min(boushey_nona_c$year):max(boushey_nona_c$year)
times <- min(boushey_nona_c$year):(max(boushey_nona_c$year)+length(times))
nodes <- unique(boushey_nona_c$state)

sim_base_c <- NULL
sim_time <- NULL
for(n in nodes){
  for(t in times){
    row_n <- which(boushey_nona_c$state==n)
    row_nt <- sample(rep(row_n,2),1)
    sim_base_c <- rbind(sim_base_c,boushey_nona_c[row_nt,])
    sim_time <- c(sim_time,t)
  }
}

sim_base_c$sim_time <- sim_time-min(times) +1

simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

simulated_cascades[[c]] <- simulated_cascade

}

sim_boushey <- do.call('rbind',simulated_cascades)

print("finished cascade simulation")

neha_boushey_sim <- neha(sim_boushey,node="state",time="sim_time",event="event",cascade="billname",covariates=covariates,ncore=20)

edges_subset <- neha_boushey_sim[[2]]

#precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

#print(precis)

recall <- c(recall,length(edges_subset))

print(recall)

save(list=c("edges_subset","neha_boushey_sim"),file=paste("./output_data/neha_boushey_sim_subset_null",s,".RData",sep=""))

print(s)

}







