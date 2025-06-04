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

source('replication.R')


dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages('readstata13', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('glmulti', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('combinat', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('glmnet', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('doParallel', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('foreach', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('boot', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
#install.packages('fastglm', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('MLmetrics', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )
install.packages('fastDummies', Sys.getenv('R_LIBS_USER'), repos = 'https://cloud.r-project.org/' )

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'), repos = NULL, type='source')

library(readstata13,lib.loc=Sys.getenv('R_LIBS_USER'))
library(neha,lib.loc=Sys.getenv('R_LIBS_USER'))
library(foreach,lib.loc=Sys.getenv('R_LIBS_USER'))
library(doParallel,lib.loc=Sys.getenv('R_LIBS_USER'))
library(MLmetrics,lib.loc=Sys.getenv('R_LIBS_USER'))
library(fastDummies,lib.loc=Sys.getenv('R_LIBS_USER'))



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


boehmke_nona <- na.omit(boehmke.p2[,c(covariates,"adopt","statepol","policy","state")])

state_dummies <- dummy_cols(boehmke_nona$state)
state_dummies <- state_dummies[,-1]
names(state_dummies) <- substr(names(state_dummies),7,nchar(names(state_dummies)))

covariates <- c(covariates,names(state_dummies)[-1])

boehmke_nona<- data.frame(boehmke_nona,state_dummies)

library(neha)

est0 <- glm(boehmke_nona$adopt ~ as.matrix(boehmke_nona[,covariates]),family=binomial)

covariates <- covariates[which(!is.na(coef(est0)[-1]))]

load("./output_data/neha_boehmke_result.RData")

neha_formula <- as.formula(paste("adopt~",paste(c(covariates,"edge_sum"),collapse="+"),sep=""))

edge_effs  <- neha_boehmke[[2]]

neha_boehmke[[3]]$edge_sum <- apply(neha_boehmke[[3]][,edge_effs],1,sum)

neha_coef <- coef(glm(neha_formula,family=binomial,data=neha_boehmke[[3]]))

# simulate a cascade
ucascades <- unique(boehmke_nona$policy)
boehmke_nona$Intercept <- 1

beta <- cbind(neha_coef[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(neha_coef[-(1:(length(covariates)+1))],length(edge_effs)))
rownames(gamma) <- edge_effs

a_par <- neha_boehmke[[1]]

# simulate data

nsim <- 50
set.seed(9202016)

precis <- NULL
recall <- NULL

for(s in 1:nsim){

  sim_boehmke <-NULL

  simulated_cascades <- list()
  for(c  in 1:length(ucascades)){

    boehmke_nona_c <- boehmke_nona[which(boehmke_nona$policy==ucascades[c]),]




    times <- min(boehmke_nona_c$time):max(boehmke_nona_c$time)
    times <- min(boehmke_nona_c$time):(max(boehmke_nona_c$time)+length(times))
    nodes <- unique(boehmke_nona_c$state)

    sim_base_c <- NULL
    sim_time <- NULL
    for(n in nodes){
      for(t in times){
        row_n <- which(boehmke_nona_c$state==n)
        row_nt <- sample(rep(row_n,2),1)
        sim_base_c <- rbind(sim_base_c,boehmke_nona_c[row_nt,])
        sim_time <- c(sim_time,t)
      }
    }

    sim_base_c$sim_time <- sim_time-min(times) +1

    simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

    simulated_cascades[[c]] <- simulated_cascade

  }

  sim_boehmke <- do.call('rbind',simulated_cascades)

  print("finished cascade simulation")

  neha_boehmke_sim <- neha(sim_boehmke,node="state",time="sim_time",event="event",cascade="policy",covariates=covariates,ncore=20)

  edges_subset <- neha_boehmke_sim[[2]]

  precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

  print(precis)

  recall <- c(recall,mean(is.element(edge_effs,edges_subset)))

  print(recall)

  save(list=c("edges_subset","neha_boehmke_sim"),file=paste("./output_data/neha_boehmke_sim_subset",s,".RData",sep=""))

  print(s)

}














