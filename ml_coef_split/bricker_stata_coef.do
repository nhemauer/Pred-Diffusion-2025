***************************************************************
* Step 0. Define log file (output .txt)
***************************************************************
cap which esttab
if _rc {
    ssc install estout
}

cd "/storage/work/ndh5286/Projects/Pred_Diffusion_2025"

capture log close
log using "ml_coefficient_split/figures/bricker_lacombe2021/bricker_split_results.txt", replace text

***************************************************************
* Step 1. Split the dataset into halves by year
***************************************************************
use "data/bricker_lacombe2021.dta", clear

* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 2. Run models and store results
***************************************************************
* First (earlier years)
melogit adoption std_score initiative init_sigs ///
    std_pop std_citideology unified std_income std_legp_squire ///
    duration durationsq durationcb i.year if sample_half == 1 || policyno:
estimates store first

* Last (later years)
melogit adoption std_score initiative init_sigs ///
    std_pop std_citideology unified std_income std_legp_squire ///
    duration durationsq durationcb i.year if sample_half == 2 || policyno:
estimates store last

***************************************************************
* Step 3. Save and Export Models
***************************************************************

* Export coefficient tables to CSV
esttab first last using "ml_coefficient_split/figures/bricker_lacombe2021/bricker_stata_results.csv", replace ///
    cells("b se") label
    
log close