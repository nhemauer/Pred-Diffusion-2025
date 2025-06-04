# 1. ### summarize predictive performance ###

load('../bricker_lacombe_analysis/output_data/neha_bl_result.RData')

library(neha)

bl_nona <- neha_bl[[3]]

edge_effs <- neha_bl[[2]]

edge_list <- do.call('rbind', strsplit(edge_effs,"_"))

library(sna)

ustates <- sort(unique(bl_nona$state))

amat <- matrix(0,length(ustates),length(ustates))

colnames(amat) <- rownames(amat) <- ustates

amat[edge_list] <- 1


yrs <- sort(unique(bl_nona$year))

prof_mats <- list()

for(i in 1:length(yrs)){
  prof_mat <- matrix(0,length(ustates),length(ustates))
  dati <- bl_nona[which(bl_nona$year==yrs[i]), ]
  for(j in 1:length(ustates)){
    prof_mat[j, ] <- dati$std_legp_squire[min(which(dati$state==ustates[j]))]
  }
  prof_mats[[i]] <- prof_mat
}

set.seed(10042016)

outdegree_centralization <- function(amat){
  require(sna)
  centralization(amat,degree,cmode="outdegree")
}

outdegree_gini <- function(amat){
  require(sna)
  require(DescTools)
  Gini(degree(amat,cmode="outdegree"))
}


ct_results <- cug.test(amat,outdegree_gini, cmode="dyad.census",reps=50000)

library(ggplot2)
df <- data.frame(simulated_gini = ct_results$rep.stat)

p<-ggplot(df, aes(x=simulated_gini)) + 
  geom_histogram(color="black", fill="white",bins=25) + xlab("Gini Coefficient")
p+ geom_vline(xintercept=ct_results$obs.stat,
              color="blue", linetype="dashed",size=1.5)
                                                          
ggsave("bl_gini_cugtest.pdf", width = 4, height = 3)

#' @article{hu2005gini,
#'   title={The Gini coefficient's application to general complex networks},
#'   author={Hu, Hai-Bo and Wang, Lin},
#'   journal={Advances in complex systems},
#'   volume={8},
#'   number={01},
#'   pages={159--167},
#'   year={2005},
#'   publisher={World Scientific}
#' }


set.seed(10042016)
resmat <- matrix(NA,length(prof_mats),6)
qap_ests <- list()
for(t in c(1,12,length(prof_mats))){
  netl_est <- netlogit(amat,list(prof_mats[[t]],t(prof_mats[[t]]), abs(prof_mats[[t]]- t(prof_mats[[t]]))),reps=2000)
  qap_ests[[t]] <- netl_est
  resmat[t,c(1,3,5)] <- netl_est$coefficients[2:4]
  resmat[t,c(2,4,6)] <- netl_est$pgreqabs[2:4]
  print(t)
}

rownames(resmat) <- yrs
colnames(resmat) <- c("send.b","send.p","rec.b","rec.p","dif.b","dif.p")

library(xtable)
qap_xt <- xtable(na.omit(resmat),digits=4)
print(qap_xt,file="bl_qap.tex")

save(list="qap_ests",file="bl_qapres.RData")


