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

import delimited "data/mallinson_lovell2022.csv", clear

. destring, replace

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
logit adopt republican legprof_squire exp_pupil10000_adj ///
    mathscore4th readscore4th time ///
    if sample_half == 1, vce(cluster state)
estimates store first50

* Second half
logit adopt republican legprof_squire exp_pupil10000_adj ///
    mathscore4th readscore4th time ///
    if sample_half == 2, vce(cluster state)
estimates store last50

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.year) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(3) xrescale) ///
    rename(republican = "Republican" legprof_squire = "Legislative Professionalism" exp_pupil10000_adj = "Net Expenditures Per Pupil" readscore4th = "Reading" mathscore4th = "Math" time = "Time") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_coefficient_split/figures/mallinson_lovell2022/mallinson_lovell_coefplot_split.png", replace width(2000)