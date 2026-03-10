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

use "data/bricker_lacombe2021.dta", clear

* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

* First half
melogit adoption std_score initiative init_sigs ///
    std_pop std_citideology unified std_income std_legp_squire ///
    duration durationsq durationcb i.year if sample_half == 1 || policyno:
estimates store first50

* Second half
melogit adoption std_score initiative init_sigs ///
    std_pop std_citideology unified std_income std_legp_squire ///
    duration durationsq durationcb i.year if sample_half == 2 || policyno:
estimates store last50

* Create coefplot
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.year) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    sort(, descending) ///
    rename(std_score = "Similarity" initiative = "Initiative Process" init_sigs = "Average Signatures" ///
           std_population = "Population" std_citideology = "Citizen Ideology" unified = "Unified Control" std_income = "Income" std_legp_squire = "Legislative Professionalism" ///
           duration = "Duration" durationsq = "Duration Squared" durationcb = "Duration Cubed") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_covariate_analysis/ml_coef_split/figures/bricker_lacombe2021/bricker_coefplot_split.png", replace width(2000)