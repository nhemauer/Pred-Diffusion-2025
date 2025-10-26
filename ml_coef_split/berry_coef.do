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

import delimited "data/berry_berry1990_processed.csv", clear

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
logit adopt fiscal_1 party elect1 elect2 income_1 nbrpercn religion if sample_half == 1,
estimates store first50

* Second half
logit adopt fiscal_1 party elect1 elect2 income_1 nbrpercn religion if sample_half == 2,
estimates store last50

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons) nolabel xline(0) ///
    rename(fiscal_1 = "Fiscal" party = "Party" elect1 = "Elect1" elect2 = "Elect2" income_1 = "Income" nbrpercn = "Neighbors" religion = "Religion") ///
    xtitle("Logit Coefficients")

graph export "ml_coefficient_split/figures/berry_berry1990/berry_coefplot_split.png", replace width(2000)