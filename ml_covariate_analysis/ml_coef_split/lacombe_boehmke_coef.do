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

use "data/lacombe_boehmke2021.dta", clear

* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

* First half
melogit adoption initiative init_sigs std_latnt_decay std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year ///
    if sample_half == 1 || policyno: std_masssociallib_est, covariance(un)
estimates store first50

* Second half
melogit adoption initiative init_sigs std_latnt_decay std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year ///
    if sample_half == 2 || policyno: std_masssociallib_est, covariance(un)
estimates store last50

* Create coefplot
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons *.year) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    sort(, descending) ///
    rename(initiative = "Initiative Process" init_sigs = "Signatures" std_latnt_decay = "Latent Decay" std_nbrs_lag = "Contiguity" std_population = "Population" ///
           std_masssociallib_est = "Public Liberalism" unified = "Unified Control" duration = "Duration" durationsq = "Duration Squared" durationcb = "Duration Cubed" ///
           std_income = "Income per Capita" std_bowen_1 = "Legislative Prof. Dim. 1" std_bowen_2 = "Legislative Prof. Dim. 2" change_pop = "Change Population" change_inc = "Change Income" ///
           party_change = "Change in Party") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_covariate_analysis/ml_coef_split/figures/lacombe_boehmke2021/lacombe_coefplot_split.png", replace width(2000)