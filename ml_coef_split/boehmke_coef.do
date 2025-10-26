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

use "data/boehmke2017.dta", clear

***************************************************************
* Step 2. Keep variables of interest and drop missing
***************************************************************
keep state year statepol adopt srcs_decay nbrs_lag rpcpinc totpop legp_squire ///
     citi6010 unif_rep unif_dem time time_sq time_cube
drop if missing(adopt, srcs_decay, nbrs_lag, rpcpinc, totpop, legp_squire, ///
                citi6010, unif_rep, unif_dem, time, time_sq, time_cube)

***************************************************************
* Step 3. Sort by year and create split indicator (first 50% vs last 50%)
***************************************************************
* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 4. Create dummy variables for state (excluding one)
***************************************************************
tabulate state, generate(state_)
drop state_1

***************************************************************
* Step 5. Run logistic regressions for each split
***************************************************************
* Fit first half
logit adopt srcs_decay nbrs_lag rpcpinc totpop legp_squire citi6010 ///
      unif_rep unif_dem time time_sq time_cube state_* ///
      if sample_half == 1, vce(cluster statepol)
estimates store first50

* Fit second half
logit adopt srcs_decay nbrs_lag rpcpinc totpop legp_squire citi6010 ///
      unif_rep unif_dem time time_sq time_cube state_* ///
      if sample_half == 2, vce(cluster statepol)
estimates store last50

***************************************************************
* Step 6. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons state_*) nolabel xline(0) ///
    rename(srcs_decay = "Lag Source Adoptions" nbrs_lag = "Lag Neighbor Adoptions" rpcpinc = "Personal Income" totpop = "Total Population" legp_squire = "Legislative Professionalism" ///
           citi6010 = "State Citizen Ideology" unif_rep = "Unified Republican Control" unif_dem = "Unified Democratic Control" time = "Time" time_sq = "Time Squared" time_cube = "Time Cubed") ///
    xtitle("Logit Coefficients")

graph export "ml_coefficient_split/figures/boehmke2017/boehmke_coefplot_split.png", replace width(2000)