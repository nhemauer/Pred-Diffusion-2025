### Packages ###
library(foreign)
library(readstata13)
library(sandwich)
library(lmtest)
library(rms)
library(xtable)
library(neha)

load('../karch_analysis/results_summaries/minyr_estimate.RData')

karch_res <- neha.yrfun

load('../boushey_analysis/results_summaries/minyr_estimate.RData')

boushey_res <- neha.yrfun

load('../boehmke_analysis/results_summaries/minyr_estimate.RData')

boehmke_res <- neha.yrfun

m2_df <- coef(summary(m2)) %>% 
  data.frame() %>% 
  add_rownames("term") %>%
  rename(estimate = Estimate, std.error = Std..Error)
m2_df

df0 <- data.frame(term = c('Boehmke','Boushey','Karch'),estimate= c(boehmke_res$coefficients['minyr'], boushey_res$coefficients['minyr'],karch_res$coefficients['minyr']), std.error= sqrt( c(boehmke_res$var['minyr','minyr'], boushey_res$var['minyr','minyr'],karch_res$var['minyr','minyr'])  ) )

library(dotwhisker)
dwplot(df0)  + geom_vline(xintercept = 0, linetype="dashed")
ggsave("minyr_coef.pdf", width = 4, height = 3)






