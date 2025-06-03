################################################################################
# "A Diffusion Network Event History Estimator"                                #
#                                                                              #
# Jeff Harden, Mark Brockway, Fred Boehmke, Bruce Desmarais, Scott LaCombe,    #
# Fridolin Linder, and Hanna Wallach                                           #
#                                                                              #
# Replication analyses                                                         #
# Last update: 6/23/22                                                         #
################################################################################
### Packages ###
### Packages ###
Sys.sleep(240)

source('replication.R')

dir.create(Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)

install.packages("foreach", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("doRNG", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("combinat", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/", INSTALL_opts = '--no-lock' )
install.packages("glmulti", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("doParallel", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("readstata13", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("MLmetrics", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')
install.packages("haven", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')

install.packages("fastglm", Sys.getenv("R_LIBS_USER"), repos = "https://mirror.las.iastate.edu/CRAN/" , INSTALL_opts = '--no-lock')

install.packages('neha_0.1.0.tar.gz', Sys.getenv('R_LIBS_USER'),repos = NULL, type='source', INSTALL_opts = '--no-lock')

library(foreach,lib.loc=Sys.getenv("R_LIBS_USER"))
library(doParallel,lib.loc=Sys.getenv("R_LIBS_USER"))
library(readstata13,lib.loc=Sys.getenv("R_LIBS_USER"))
library(MLmetrics,lib.loc=Sys.getenv("R_LIBS_USER"))
library(haven,lib.loc=Sys.getenv("R_LIBS_USER"))
library(neha,lib.loc=Sys.getenv("R_LIBS_USER"))
library(doRNG,lib.loc=Sys.getenv("R_LIBS_USER"))

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


## BL data ##
bl <- read_dta("./replication_data/monadic_analysis_largen.dta")

# melogit adoption std_score initiative init_sigs std_pop std_citideology unified std_income std_legp_squire duration  durationsq durationcb i.year || policyno:

options(na.action='na.pass')
data_for_bl <- model.matrix(adoption ~ std_score + std_score + initiative + init_sigs + std_population + std_citideology + unified + std_income + std_legp_squire + duration+ durationsq + durationcb + as.factor(year),data=bl)[,-1]

colnames(data_for_bl) <- gsub("as.factor\\(year\\)","y",colnames(data_for_bl))

covariates <- colnames(data_for_bl)

data_for_bl <- data_for_bl[,-length(covariates)]
covariates <- covariates[-length(covariates)]


data_for_bl <- data.frame(data_for_bl)
data_for_bl[,c("adoption","state","year","policyno")] <- bl[,c("adoption","state","year","policyno")]

data_for_bl$policyno <- paste("p",data_for_bl$policyno,sep="")

bl_nona <- na.omit(data_for_bl)

library(neha)

load("./output_data/neha_bl_result.RData")

neha_formula <- as.formula(paste("adoption~",paste(c(covariates,"edge_sum"),collapse="+"),sep=""))

edge_effs  <- neha_bl[[2]]

neha_bl[[3]]$edge_sum <- apply(neha_bl[[3]][,edge_effs],1,sum)

neha_coef <- coef(glm(neha_formula,family=binomial,data=neha_bl[[3]]))

# simulate a cascade
ucascades <- unique(bl_nona$policyno)
bl_nona$Intercept <- 1

beta <- cbind(neha_coef[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(0,length(edge_effs)))
rownames(gamma) <- edge_effs

a_par <- neha_bl[[1]]

# simulate data

nsim <- 50
set.seed(9202016)

precis <- NULL
recall <- NULL

cl <- makeCluster(10)
registerDoParallel(cl)

for(s in 1:nsim){

  sim_bl <- NULL

  simulated_cascades <- list()
  for(c  in 1:length(ucascades)){

    bl_nona_c <- bl_nona[which(bl_nona$policyno==ucascades[c]),]


    times <- min(bl_nona_c$year):max(bl_nona_c$year)
    times <- min(bl_nona_c$year):(max(bl_nona_c$year)+length(times))
    nodes <- unique(bl_nona_c$state)

    sim_base_c <- NULL
    sim_time <- NULL
    for(n in nodes){
      for(t in times){
        row_n <- which(bl_nona_c$state==n)
        row_nt <- sample(rep(row_n,2),1)
        sim_base_c <- rbind(sim_base_c,bl_nona_c[row_nt,])
        sim_time <- c(sim_time,t)
      }
    }

    sim_base_c$sim_time <- sim_time-min(times) +1

    simulated_cascade <- simulate_neha_discrete(sim_base_c,"state","sim_time",beta,gamma,a_par)

    simulated_cascades[[c]] <- simulated_cascade

  }

  sim_bl <- do.call('rbind',simulated_cascades)

  print("finished cascade simulation")
  
  ## WORKAROUND: https://github.com/rstudio/rstudio/issues/6692
  ## Revert to 'sequential' setup of PSOCK cluster in RStudio Console on macOS and R 4.0.0
  if (Sys.getenv("RSTUDIO") == "1" && !nzchar(Sys.getenv("RSTUDIO_TERM")) && 
      Sys.info()["sysname"] == "Darwin" && getRversion() >= "4.0.0") {
    parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
  }

  neha_bl_sim <- neha(sim_bl,node="state",time="sim_time",event="event",cascade="policyno",covariates=covariates,ncore=18)

  edges_subset <- neha_bl_sim[[2]]

  precis <- c(precis,mean(is.element(edges_subset,edge_effs)))

  print(precis)

  recall <- c(recall,mean(is.element(edge_effs,edges_subset)))

  print(recall)

  save(list=c("edges_subset","neha_bl_sim"),file=paste("./output_data/neha_bl_sim_subset_zero",s,".RData",sep=""))

  print(s)

}


stopCluster(cl)











