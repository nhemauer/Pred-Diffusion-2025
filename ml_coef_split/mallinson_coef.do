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

import delimited "data/mallinson2019_processed.csv", clear

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
melogit adopt neighbor_prop ideology_relative_hm congress_majortopic ///
    init_avail init_qual divided_gov legprof_squire ///
    percap_log population_log mip complexity_topic ///
    mip_complexity_topic nyt year_count time_log ///
    if sample_half == 1 || policy: neighbor_prop,
estimates store first50

* Second half
melogit adopt neighbor_prop ideology_relative_hm congress_majortopic ///
    init_avail init_qual divided_gov legprof_squire ///
    percap_log population_log mip complexity_topic ///
    mip_complexity_topic nyt year_count time_log ///
    if sample_half == 2 || policy: neighbor_prop,
estimates store last50

* Full
melogit adopt neighbor_prop ideology_relative_hm congress_majortopic ///
    init_avail init_qual divided_gov legprof_squire ///
    percap_log population_log mip complexity_topic ///
    mip_complexity_topic nyt year_count time_log ///
    || policy: neighbor_prop, cov(un)

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.year) nolabel xline(0) ///
    rename(neighbor_prop = "Neighbor Adoptions" ideology_relative_hm = "Ideological Distance" congress_majortopic = "Congressional Hearings" init_avail = "Iniative Available" init_qual = "Iniative Qual. Difficulty" ///
           divided_gov = "Divided Government" legprof_squire = "Legislative Professionalism" percap_log = "Per Capita Income" population_log = "Population" mip = "Most Important Problem" complexity_topic = "Complex Policy" ///
           mip_complexity_topic = "MIP x Complex" nyt = "New York Times" year_count = "Year" time_log = "Time") ///
    xtitle("Logit Coefficients")

graph export "ml_coefficient_split/figures/mallinson2019/mallinson_coefplot_split.png", replace width(2000)