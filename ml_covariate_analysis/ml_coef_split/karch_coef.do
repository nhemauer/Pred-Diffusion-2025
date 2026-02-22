***************************************************************
* Step 1. Load data
***************************************************************

cap which coefplot
if _rc {
    ssc install coefplot
}

cap which blindschemes
if _rc {
    ssc install blindschemes
}

set scheme plotplain

cd "/storage/work/ndh5286/Projects/Pred_Diffusion_2025"

use "data/karch2016.dta", clear

***************************************************************
* Step 2. Sort by year and create split indicator (first 50% vs last 50%)
***************************************************************
* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 3. Run logistic regressions for each split
***************************************************************
* First half
logit adopt traditional nborsstd prevadoptstd complexity igrole ///
      regov unified perdemstd incpcadjstd exppcadjstd ///
      logpopstd collegstd perurbanstd profstd ///
      traditional_nborsstd traditional_prevadoptstd traditional_complexity ///
      traditional_igrole traditional_regov traditional_unified ///
      traditional_perdemstd traditional_incpcadjstd traditional_exppcadjstd ///
      traditional_logpopstd traditional_collegstd traditional_perurbanstd traditional_profstd ///
      if sample_half == 1, vce(cluster stateyear)
estimates store first50

* Second half
logit adopt traditional nborsstd prevadoptstd complexity igrole ///
      regov unified perdemstd incpcadjstd exppcadjstd ///
      logpopstd collegstd perurbanstd profstd ///
      traditional_nborsstd traditional_prevadoptstd traditional_complexity ///
      traditional_igrole traditional_regov traditional_unified ///
      traditional_perdemstd traditional_incpcadjstd traditional_exppcadjstd ///
      traditional_logpopstd traditional_collegstd traditional_perurbanstd traditional_profstd ///
      if sample_half == 2, vce(cluster stateyear)
estimates store last50

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.year) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    rename(traditional = "Traditional" nborsstd = "Neighbors" prevadoptstd = "Previous Adopters" complexity = "Complexity" igrole = "Interest Group Role" ///
           regov = "Republican Governor" unified = "Unified" perdemstd = "Democratic Legislature" incpcadjstd = "Income per Capita" exppcadjstd = "Expenditures per Capita" ///
           logpopstd = "Population" collegstd = "Pct College Educated" perurbanstd = "Pct Urban" profstd = "Legislative Professionalism" ///
           traditional_nborsstd = "Traditional x Neighbors" traditional_prevadoptstd = "Traditional x Prev. Adopters" traditional_complexity = "Traditional x Complexity" ///
           traditional_igrole = "Traditional x Interest Group" traditional_regov = "Traditional x Rep. Governor" traditional_unified = "Traditional x Unified" ///
           traditional_perdemstd = "Traditional x Dem. Legislature" traditional_incpcadjstd = "Traditional x Income" traditional_exppcadjstd = "Traditional x Expenditures" ///
           traditional_logpopstd = "Traditional x Population" traditional_collegstd = "Traditional x College" traditional_perurbanstd = "Traditional x Urban" traditional_profstd = "Traditional x Professionalism") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_covariate_analysis/ml_coef_split/figures/karch2016/karch_coefplot_split.png", replace width(2000)