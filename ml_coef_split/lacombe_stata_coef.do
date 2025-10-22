***************************************************************
* Step 0. Define log file (output .txt)
***************************************************************
cap which esttab
if _rc {
    ssc install estout
}

cd "/storage/work/ndh5286/Projects/Pred_Diffusion_2025"

capture log close
log using "ml_coefficient_split/figures/lacombe_boehmke2021/lacombe_split_results.txt", replace text

***************************************************************
* Step 1. Split the dataset into halves by year
***************************************************************
use "data/lacombe_boehmke2021.dta", clear

* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 2. Run models and store results
***************************************************************
* First (earlier years)
melogit adoption initiative init_sigs std_latnt_decay std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year if sample_half == 1 ///
    || policyno: std_masssociallib_est, cov(un)
estimates store first

* Last (later years)
melogit adoption initiative init_sigs std_latnt_decay std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year if sample_half == 2 ///
    || policyno: std_masssociallib_est, cov(un)
estimates store last

***************************************************************
* Step 3. Save and Export Models
***************************************************************
* Export coefficient tables to CSV
esttab first last using "ml_coefficient_split/figures/lacombe_boehmke2021/lacombe_stata_results.csv", replace ///
    cells("b se") label
    
log close