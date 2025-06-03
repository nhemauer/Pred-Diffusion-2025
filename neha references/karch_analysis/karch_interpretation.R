# 1. ### summarize predictive performance ###

load('karch_oos_results.RData')
load('neha_karch_result.RData')


library(MLmetrics)

res_file <- dir()
res_file <- res_file[grepl("karch_subset_oos_predict"
                           ,res_file)]

load(res_file[[1]])
y_test <- test_data[,"adopt"]

testy <- NULL
peha_scorey <- NULL
neha_scorey <- NULL
neha_scorey_s <- NULL
policy_id <- NULL

for(r in res_file){
  load(r)
  y_test <- test_data[,"adopt"]
  testy <- c(testy,y_test)
  peha_scorey <- c(peha_scorey,peha.test)
  neha_scorey <- c(neha_scorey,neha.test)
  neha_scorey_s <- c(neha_scorey_s,neha.test.s)
  policy_id <- c(policy_id,rep(r,nrow(test_data)))
}

pr_peha <- PRAUC(peha_scorey, testy)
pr_neha <- PRAUC(neha_scorey, testy)
pr_neha_s <- PRAUC(neha_scorey_s, testy)

# cluster-bootstrap samples


set.seed(10042016)
nboot <- 500

peha_boot <- numeric(nboot)
neha_boot <- numeric(nboot)
neha_s_boot <- numeric(nboot)

for(i in 1:nboot){
  sampi <- sample(res_file,length(res_file),rep=T)
  peha_scorey_i <- NULL
  neha_scorey_i <- NULL
  neha_scorey_s_i <- NULL
  testy_i <- NULL
  for(j in 1:length(sampi)){
    peha_scorey_i <- c(peha_scorey_i, peha_scorey[which(policy_id==sampi[j])])
    neha_scorey_i <- c(neha_scorey_i, neha_scorey[which(policy_id==sampi[j])])
    neha_scorey_s_i <- c(neha_scorey_s_i, neha_scorey_s[which(policy_id==sampi[j])])
    testy_i <- c(testy_i, testy[which(policy_id==sampi[j])])
  }
  
  peha_boot[i] <- PRAUC(peha_scorey_i, testy_i)
  neha_boot[i] <- PRAUC(neha_scorey_i, testy_i)
  neha_s_boot[i] <- PRAUC(neha_scorey_s_i, testy_i)
  
  if(i/10 == round(i/10)) print(i)
}

p_peha_neha <- 2*min(c(mean(peha_boot-neha_boot > 0), mean(peha_boot-neha_boot < 0)))
p_peha_neha_s <- 2*min(c(mean(peha_boot-neha_s_boot > 0), mean(peha_boot-neha_s_boot < 0)))
p_neha_neha_s <- 2*min(c(mean(neha_boot-neha_s_boot > 0), mean(neha_boot-neha_s_boot < 0)))

save(list=c("peha_boot","neha_boot","neha_s_boot"),file="fit_boot.RData")



library(MLmetrics)

policy <- neha_karch[[3]]$policy
upol <- unique(policy)
pr_peha <- numeric(length(upol))
pr_neha <- numeric(length(upol))
pr_neha_s <- numeric(length(upol))

start <- 1

for(p in 1:length(upol)){
  
  nobs <- length(which(policy==upol[p]))
  end <- start + nobs - 1
  
  if(mean(testy[start:end]) > 0){
  
  pr_peha[p] <- PRAUC(peha_scorey[start:end], testy[start:end])
  pr_neha[p] <- PRAUC(neha_scorey[start:end], testy[start:end])
  pr_neha_s[p] <- PRAUC(neha_scorey_s[start:end], testy[start:end])
  
  }
  
  start <- start + nobs
}

library(xtable)

fitmat <- rbind(c(mean(pr_peha),mean(pr_neha),mean(pr_neha_s)))

rownames(fitmat) <- c("AUC-PR")
colnames(fitmat) <- c("PEHA","NEHA","NEHA-S")

xtable(fitmat,dig=4)

# 2. ### Model Estimation ###

### Packages ###
library(foreign)
library(readstata13)
library(sandwich)
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


## Karch et al. (2016) ##
karch <- read.dta("karch2016.dta")
karch <- karch[order(karch$state, karch$year), ] 
karch[is.na(karch$stateyear), ]$stateyear <- 10

#compnum

# Table 2, column 1 (90) #
peha.karch <- robcov(lrm(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd, x = TRUE, y = TRUE, data = karch), cluster = karch$stateyear)

options(na.action='na.pass')
data_for_karch <- model.matrix(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd,data=karch)[,-1]

colnames(data_for_karch) <- gsub(":","_",colnames(data_for_karch))

covariates <- colnames(data_for_karch)


data_for_karch <- data.frame(data_for_karch)
data_for_karch[,c("adopt","state","year","stateyear")] <- karch[,c("adopt","state","year","stateyear")]

data_for_karch$policy <- paste("p",karch$compnum,sep="")

karch_nona <- na.omit(data_for_karch)


library(neha)

karch_nona <- neha_karch[[3]]

edge_effs <- neha_karch[[2]]

neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,edge_effs),collapse="+")))


neha_formula <- as.formula(paste("adopt ~",paste(c(covariates,"edge_sum"),collapse="+")))

karch_nona$edge_sum <- apply(karch_nona[,edge_effs],1,sum)

neha.karch <- robcov(lrm(neha_formula, x = TRUE, y = TRUE, data = karch_nona), cluster = karch_nona$stateyear)

#  edge_sum                  2.0710 0.1455  14.23 <0.0001 
#  edge_sum                  2.0660 0.1452  14.23 <0.0001 
#  edge_sum                  2.0677 0.1453  14.23 <0.0001 
#  edge_sum                  2.0677 0.1453  14.23 <0.0001 
#  edge_sum                  2.0677 0.1453  14.23 <0.0001
#  edge_sum                  2.0677 0.1453  14.23 <0.0001 

mean_lp <- mean(cbind(1,neha.karch$x)%*%cbind(coef(neha.karch)))
times <- 1:50
alpha_fun <- function(time,alpha){
  exp(-alpha*time)
}

library(ggplot2)
df <- data.frame(years = times,diffusion_effect=plogis(mean_lp + coef(neha.karch)[length(coef(neha.karch))]*alpha_fun(times,exp(neha_karch[[1]]))))
ggplot(data=df, aes(x=times, y=diffusion_effect, group=1)) +
  geom_line() + 
  ylim(0,.10)+
  xlim(min(times),max(times))+
  xlab("yrs since source adoption")+
  ylab("Pr(adoption)") +
  geom_hline(aes(yintercept=plogis(mean_lp)),linetype="dashed")
ggsave("karch_decay.pdf", width = 4, height = 3)

neha.bic <- BIC(glm(neha_formula, family=binomial,data=karch_nona))

peha.bic <- BIC(glm(adopt ~ traditional + nborsstd +  traditional*nborsstd+prevadoptstd + traditional*prevadoptstd + complexity + traditional*complexity + igrole + traditional*igrole + regov + traditional*regov + unified + traditional*unified + perdemstd + traditional*perdemstd + incpcadjstd + traditional*incpcadjstd + exppcadjstd + traditional*exppcadjstd + logpopstd + traditional*logpopstd + collegstd + traditional*collegstd + perurbanstd + traditional*perurbanstd + profstd + traditional*profstd, family=binomial,data=karch_nona))

reg_table <- cbind(c(coef(peha.karch),NA), c(sqrt(diag(peha.karch$var)),NA) , c(coef(neha.karch)), c(sqrt(diag(neha.karch$var))))
reg_table <- rbind(reg_table,c(NA,NA,NA,exp(neha_karch[[1]])))
reg_table <- rbind(reg_table,c(nrow(karch_nona),length(upol),NA,NA))
reg_table <- rbind(reg_table,c(NA,peha.bic,NA,neha.bic))

rownames(reg_table)[c(nrow(reg_table)-3,nrow(reg_table)-2,nrow(reg_table)-1,nrow(reg_table))] <- c("gamma","alpha","obs_policy","BIC")

xtable(reg_table,dig=4)


edge_list <- do.call('rbind', strsplit(edge_effs,"_"))

library(igraph)

net <- graph_from_edgelist(edge_list)

set.seed(9202011)
pdf("karch_edge_plot.pdf",height=4,width=4,pointsize=8)
plot(net, edge.arrow.size=.4, edge.color="blue", vertex.color="grey75", vertex.frame.color="grey75", vertex.label.color="black") 
dev.off()



# 3. ### simulation performance summary ###
# precision, recall, and frequency table of 

precision <- NULL
recall <- NULL
n_null <- NULL

sim_edges <- list()
sim_edges_zero <- list()

gamma <- numeric(50)
gamma_zero <- numeric(50)
neha_formula <-  as.formula(paste("event ~",paste(c(covariates,"edge_sum"), collapse="+" ),sep=""))

for(s in 1:50){
  
  load(paste("neha_karch_sim_subset",s,".RData",sep=""))
  sim_edges[[s]] <- edges_subset
  precision <- c(precision, mean(is.element(edges_subset,edge_effs)))
  neha_karch_sim[[3]]$edge_sum <- apply(neha_karch_sim[[3]][,edges_subset],1,sum)
  
  est_neha_sim <- glm(neha_formula,family=binomial,data=neha_karch_sim[[3]])
  
  gamma[s] <- coef(est_neha_sim)[length(coef(est_neha_sim))]
  
  recall <- c(recall, mean(is.element(edge_effs,edges_subset)))
  load(paste("neha_karch_sim_subset_zero",s,".RData",sep=""))
  sim_edges_zero[[s]] <- edges_subset
  n_null <- c(n_null,length(edges_subset))
  
  if(length(edges_subset) > 0){
    neha_karch_sim[[3]]$edge_sum <- apply(cbind(neha_karch_sim[[3]][,edges_subset]),1,sum)
    est_neha_sim <- glm(neha_formula,family=binomial,data=neha_karch_sim[[3]])
    gamma_zero[s] <- coef(est_neha_sim)[length(coef(est_neha_sim))]
  }
  
}

mean(precision)
mean(recall)


sim_res <- c(mean(precision),mean(recall),max(n_null),median(n_null))
names(sim_res) <- c("Precision","Recall","Max null","Med null")
xtable(cbind(sim_res),dig=4)


p<-ggplot(df, aes(x=gamma)) + 
  geom_histogram(color="black", fill="white",bins=10)
p+ geom_vline(aes(xintercept=coef(neha.karch)["edge_sum"]),
              color="blue", linetype="dashed", size=1) + geom_vline(aes(xintercept=mean(gamma)),
                                                                    color="red", size=1)
ggsave("karch_gamma_sim.pdf", width = 4, height = 3)







