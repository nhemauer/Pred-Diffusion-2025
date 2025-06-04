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
library(haven)
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


## BL data ##
bl <- read_dta("../bricker_lacombe_analysis/replication_data/monadic_analysis_largen.dta")

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

load("../bricker_lacombe_analysis/output_data/neha_bl_result.RData")

neha_formula <- as.formula(paste("adoption~",paste(c(covariates,"edge_sum"),collapse="+"),sep=""))

neha_formula0 <- as.formula(paste("adoption~",paste(covariates,collapse="+"),sep=""))

edge_effs  <- neha_bl[[2]]

neha_bl[[3]]$edge_sum <- apply(neha_bl[[3]][,edge_effs],1,sum)

neha_coef <- coef(glm(neha_formula,family=binomial,data=neha_bl[[3]]))

# simulate a cascade
ucascades <- unique(bl_nona$policyno)
bl_nona$Intercept <- 1

beta <- cbind(neha_coef[1:(length(covariates)+1)])
rownames(beta)[1] <- "Intercept"
gamma <- cbind(rep(neha_coef[-(1:(length(covariates)+1))],length(edge_effs)))
rownames(gamma) <- edge_effs

a_par <- neha_bl[[1]]

# simulate data

nsim <- 50

beta_sim <- matrix(NA,nsim,length(beta))
beta_sim0 <- matrix(NA,nsim,length(beta))
cor_edgesum <-  matrix(NA,nsim,length(beta)-1)

for(s in 1:nsim){
  
  load(paste("../bricker_lacombe_analysis/output_data/neha_bl_sim_subset",s,".RData",sep=""))
  neha_bl_sim[[3]]$edge_sum <- apply(neha_bl_sim[[3]][,edges_subset],1,sum)
  
  est_neha_sim <- glm(neha_formula,family=binomial,data=neha_bl_sim[[3]])
  
  est_neha_sim0 <- glm(neha_formula0,family=binomial,data=neha_bl_sim[[3]])
  
  beta_sim[s,] <- coef(est_neha_sim)[1:length(beta)]
  beta_sim0[s,] <- coef(est_neha_sim0)[1:length(beta)]
  cor_edgesum[s,] <- cor(neha_bl_sim[[3]]$edge_sum,neha_bl_sim[[3]][,covariates])
  
  print(s)
  
}


est_err <- matrix(NA,nsim,length(beta)-1)
est_err0 <- matrix(NA,nsim,length(beta)-1)

for(i in 1:nrow(cor_edgesum)){
  for(j in 1:ncol(cor_edgesum)){
    est_err[i,j] <- beta_sim[i,j+1]-beta[j+1]
    est_err0[i,j] <- beta_sim0[i,j+1]-beta[j+1]
  }
}

abs_err <- abs(est_err)
abs_err0 <- abs(est_err0)

save(list=c("beta_sim","beta_sim0","abs_err","abs_err0","cor_edgesum"),file="bl_ovsim.RData")


relative_error <- abs(c(abs_err0))/abs(c(abs_err))
relative_bias <- abs(apply(abs_err0,2,mean))/ abs(apply(abs_err,2,mean))
                                  
                 
rel_perf <- data.frame(ratio = c(relative_error,relative_bias), error = c(rep("abs error",length(relative_error)), rep("abs bias",length(relative_bias))  ))
                      
library(ggplot2)

# devtools::install_github('kongdd/Ipaper')
library(Ipaper)
library(ggplot2)

p <- ggplot(rel_perf, aes(x=error, y=ratio)) + 
  geom_boxplot2(width = 0.8, width.errorbar = 0.5) +
  geom_hline(yintercept=1,linetype = 2)
p

ggsave("bl_bias.pdf", width = 4, height = 3)


mean(relative_error > 1)
mean(relative_bias > 1)





