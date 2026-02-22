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

use "data/parinandi2020.dta", clear

***************************************************************
* Step 2. Sort by year and create split indicator (first 70% vs last 30%)
***************************************************************
summ year, meanonly
local midyear = floor(r(min) + 0.7 * (r(max) - r(min)))
gen sample_split = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 3. Run logistic regressions for each split
***************************************************************
* First half
logit oneemulation c.adagovideology c.citizenideology c.medianivoteshare i.partydecline c.squirescore ///
    c.incunemp c.pctpercapincome c.percenturban i.ugovd c.percentfossilprod c.renergyprice11 i.deregulated ///
    c.geoneighborlag c.ideoneighborlag c.premulation1 c.year c.featureyear if sample_split == 1, robust cluster(statenumber)
estimates store first70

* Second half
logit oneemulation c.adagovideology c.citizenideology c.medianivoteshare i.partydecline c.squirescore ///
    c.incunemp c.pctpercapincome c.percenturban i.ugovd c.percentfossilprod c.renergyprice11 i.deregulated ///
    c.geoneighborlag c.ideoneighborlag c.premulation1 c.year c.featureyear if sample_split == 2, robust cluster(statenumber)
estimates store last30

***************************************************************
* Step 4. Create Coefplot
***************************************************************
coefplot (first70, label("First 70%")) (last30, label("Last 30%")), ///
    drop(_cons) xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    rename(adagovideology = "Legislative Ideology" citizenideology = "Citizen Ideology" medianivoteshare = "Median Incumbent Vote Share" 1.partydecline = "Party Decline" ///
           squirescore = "Legislative Professionalism" incunemp = "Change in Unemplyoment" pctpercapincome = "Per Capita Income" percenturban = "Urban Percentage" 1.ugovd = "Unified Dem. Government" ///
           percentfossilprod = "Fossil Fuel Production" renergyprice11 = "Real Energy Price" 1.deregulated = "Deregulated" geoneighborlag = "Lagged Geographic Neighbor" ///
           ideoneighborlag = "Lagged Ideological Neighbor" premulation1 = "Prior Borrowing" year = "Year" featureyear = "Provision Year") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_covariate_analysis/ml_coef_split/figures/parinandi2020/parinandi_coefplot_split.png", replace width(2000)