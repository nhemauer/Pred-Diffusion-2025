# 1. ### summarize predictive performance ###

load('./output_data/neha_bl_result.RData')

library(MLmetrics)

res_file <- dir('./output_data/')
res_file <- res_file[grepl("bl_subset_oos_predict"
                           ,res_file)]

res_file <- sort(paste('./output_data/',res_file,sep=''))

testy <- NULL
peha_scorey <- NULL
neha_scorey <- NULL
neha_scorey_s <- NULL
policy_id <- NULL

for(r in res_file){
  load(r)
  y_test <- test_data[,"adoption"]
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


set.seed(92011)
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

save(list=c("peha_boot","neha_boot","neha_s_boot"),file="./results_summaries/fit_boot.RData")

pr_results <- c(pr_peha,pr_neha,pr_neha_s,p_peha_neha,p_peha_neha_s,p_neha_neha_s)
names(pr_results) <- c("pr_peha","pr_neha","pr_neha_s","p_peha_neha","p_peha_neha_s","p_neha_neha_s")

write.csv(pr_results,file="./results_summaries/pr_results.csv")



# 2. ### Model Estimation ###

load('./output_data/neha_bl_result.RData')

### Packages ###
library(foreign)
library(readstata13)
library(sandwich)
library(haven)
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
bl <- read_dta("./replication_data/monadic_analysis_largen.dta")



# Table 2, column 1 (90) #
options(na.action='na.pass')
data_for_bl <- model.matrix(adoption ~ std_score + std_score + initiative + init_sigs + std_population + std_citideology + unified + std_income + std_legp_squire + duration+ durationsq + durationcb + as.factor(year),data=bl)[,-1]

colnames(data_for_bl) <- gsub("as.factor\\(year\\)","y",colnames(data_for_bl))

covariates <- colnames(data_for_bl)[-ncol(data_for_bl)]

data_for_bl <- data.frame(data_for_bl)
data_for_bl[,c("adoption","state","year","policyno")] <- bl[,c("adoption","state","year","policyno")]

data_for_bl$policyno <- paste("p",data_for_bl$policyno,sep="")

bl_nona <- na.omit(data_for_bl)

library(neha)

bl_nona <- neha_bl[[3]]

edge_effs <- neha_bl[[2]]

neha_formula <- as.formula(paste("adoption ~",paste(c(covariates,"edge_sum"),collapse="+")))

bl_nona$edge_sum <- apply(bl_nona[,edge_effs],1,sum)

minyr <- numeric(nrow(bl_nona))

for(i in 1:length(minyr)){
  minyr[i] <- min(bl_nona$year[which(bl_nona$policyno==bl_nona$policyno[i])])
}

bl_nona$minyr <- minyr

write.dta(bl_nona,file="./replication_data/bricker_lacombe_neha_data.dta")

neha.bl<- glm(neha_formula, x = TRUE, y = TRUE, data = bl_nona,family="binomial")


mean_lp <- mean(cbind(neha.bl$x)%*%cbind(coef(neha.bl)))
times <- 1:50
alpha_fun <- function(time,alpha){
  exp(-alpha*time)
}

library(ggplot2)
df <- data.frame(years = times,diffusion_effect=plogis(mean_lp + coef(neha.bl)[length(coef(neha.bl))]*alpha_fun(times,exp(neha_bl[[1]]))))
ggplot(data=df, aes(x=times, y=diffusion_effect, group=1)) +
  geom_line() + 
  ylim(0,.15)+
  xlim(min(times),max(times))+
  xlab("yrs since source adoption")+
  ylab("Pr(adoption)") +
  geom_hline(aes(yintercept=plogis(mean_lp)),linetype="dashed")
ggsave("./results_summaries/bl_decay.pdf", width = 4, height = 3)



edge_list <- do.call('rbind', strsplit(edge_effs,"_"))

library(igraph)

net <- graph_from_edgelist(edge_list)

set.seed(9202011)
pdf("./results_summaries/bl_edge_plot.pdf",height=4,width=4,pointsize=6)
plot(net, edge.arrow.size=.4, edge.color="blue", vertex.color="grey75", vertex.frame.color="grey75", vertex.size=3,vertex.label.color="black") 
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
  
  load(paste("./output_data/neha_bl_sim_subset",s,".RData",sep=""))
  sim_edges[[s]] <- edges_subset
  precision <- c(precision, mean(is.element(edges_subset,edge_effs)))
  neha_bl_sim[[3]]$edge_sum <- apply(neha_bl_sim[[3]][,edges_subset],1,sum)
  
  est_neha_sim <- glm(neha_formula,family=binomial,data=neha_bl_sim[[3]])
  
  gamma[s] <- coef(est_neha_sim)[length(coef(est_neha_sim))]
  
  recall <- c(recall, mean(is.element(edge_effs,edges_subset)))
  load(paste("./output_data/neha_bl_sim_subset_zero",s,".RData",sep=""))
  sim_edges_zero[[s]] <- edges_subset
  n_null <- c(n_null,length(edges_subset))
  
  if(length(edges_subset) > 0){
    neha_bl_sim[[3]]$edge_sum <- apply(cbind(neha_bl_sim[[3]][,edges_subset]),1,sum)
    est_neha_sim <- glm(neha_formula,family=binomial,data=neha_bl_sim[[3]])
    gamma_zero[s] <- coef(est_neha_sim)[length(coef(est_neha_sim))]
  }
  
  print(s)
  
}

mean(precision)
mean(recall)


sim_res <- c(mean(precision),mean(recall),max(n_null),median(n_null))
names(sim_res) <- c("Precision","Recall","Max null","Med null")
write.csv(sim_res,"./results_summaries/sim_result_summary.csv") 



df <- data.frame(gamma)

p<-ggplot(df, aes(x=gamma)) + 
  geom_histogram(color="black", fill="white",bins=10)
p+ geom_vline(aes(xintercept=coef(neha.bl)["edge_sum"]),
              color="blue", linetype="dashed", size=1) + geom_vline(aes(xintercept=mean(gamma)),
                                                                    color="red", size=1)
ggsave("./results_summaries/bl_gamma_sim.pdf", width = 4, height = 3)







