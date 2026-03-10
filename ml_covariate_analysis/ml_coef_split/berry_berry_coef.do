cap which coefplot
if _rc {
    ssc install coefplot
}

cap which blindschemes
if _rc {
    ssc install blindschemes
}

set scheme plotplain

* Change working directory
cd "/storage/work/ndh5286/Projects/Pred_Diffusion_2025"

import delimited "data/berry_berry1990_processed.csv", clear

* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

* First half
logit adopt fiscal_1 party elect1 elect2 income_1 nbrpercn religion if sample_half == 1,
estimates store first50

* Second half
logit adopt fiscal_1 party elect1 elect2 income_1 nbrpercn religion if sample_half == 2,
estimates store last50

* Create coefplot
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(3) xrescale) ///
    sort(, descending) ///
    rename(fiscal_1 = "Fiscal" party = "Party" elect1 = "Elect1" elect2 = "Elect2" income_1 = "Income" nbrpercn = "Neighbors" religion = "Religion") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_covariate_analysis/ml_coef_split/figures/berry_berry1990/berry_coefplot_split.png", replace width(2000)