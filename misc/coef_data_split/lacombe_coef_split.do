***************************************************************
* Step 0. Define log file (output .txt)
***************************************************************
capture log close
log using "lacombe_split_results.txt", replace text

***************************************************************
* Step 1. Split the dataset into halves
***************************************************************
sort year
gen sample_half = cond(_n <= _N/2, 1, 2)

***************************************************************
* Step 2. Run models and store results
***************************************************************
* Full dataset
melogit adoption initiative init_sigs std_latnt_decay  std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year ///
    || policyno: std_masssociallib_est	, cov(un)
estimates store full

* First 50%
melogit adoption initiative init_sigs std_latnt_decay  std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year if sample_half == 1 ///
    || policyno: std_masssociallib_est	, cov(un)
estimates store first

* Last 50%
melogit adoption initiative init_sigs std_latnt_decay  std_nbrs_lag std_pop ///
    std_masssociallib_est unified duration durationsq durationcb std_income ///
    std_bowen_1 std_bowen_2 change_pop change_inc party_change i.year if sample_half == 2 ///
    || policyno: std_masssociallib_est	, cov(un)
estimates store last

***************************************************************
* Step 3. Display sample sizes
***************************************************************
display as text "Sample Sizes:"

count if sample_half == 1
local n_first = r(N)
display as text "First_50%: n = `n_first'"

count if sample_half == 2
local n_last = r(N)
display as text "Last_50%: n = `n_last'"

count
local n_full = r(N)
display as text "Full_Dataset: n = `n_full'"

***************************************************************
* Step 7. Close log file
***************************************************************
log close
display as text "Results saved to lacombe_split_results.txt"