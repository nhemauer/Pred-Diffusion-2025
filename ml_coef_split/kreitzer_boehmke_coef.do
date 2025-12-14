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

use "data/kreitzer_boehmke2016.dta", clear

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
* Fit first half
logit adopt_policy norrander_legality religadhrate initdif ///
      dem_gov uni_dem_leg fem_dem nbrspct rescaledmedincome ///
      rescaledpopsize time time2 webster i.policy_num ///
      if sample_half == 1, vce(cluster state)
estimates store first50

* Fit second half
logit adopt_policy norrander_legality religadhrate initdif ///
      dem_gov uni_dem_leg fem_dem nbrspct rescaledmedincome ///
      rescaledpopsize time time2 webster i.policy_num ///
      if sample_half == 2, vce(cluster state)
estimates store last50

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.policy_num) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    rename(norrander_legality = "Abortion Opinion" religadhrate = "Religious Adherence" initdif = "Initiative Difficulty" ///
       dem_gov = "Democratic Governor" uni_dem_leg = "Unified Dem. Legislature" fem_dem = "Democratic Women" nbrspct = "Neighbor Adoption %" ///
       rescaledmedincome = "Median Income" rescaledpopsize = "Population" time = "Time" time2 = "Time Squared" webster = "Post-Webster Indicator") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_coefficient_split/figures/kreitzer_boehmke2016/kreitzer_coefplot_split.png", replace width(2000)