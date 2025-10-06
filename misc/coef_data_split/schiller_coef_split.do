***************************************************************
* Step 0. Define log file (output .txt)
***************************************************************
capture log close
log using "schiller_split_results.txt", replace text

***************************************************************
* Step 1. Split the dataset into halves
***************************************************************
sort year
gen sample_half = cond(_n <= _N/2, 1, 2)

***************************************************************
* Step 2. Run models and store results
***************************************************************
* Full dataset
logit dvgunlaw gunhomicideslag1 citizenideologylag1 numregdvgunlawenactlag1 ///
    vawa1994 vawa1995 lautenbergamdt1996 Lautenbergamndt1997 ///
    legislature_election_year femleg innovation_index, vce(cluster state)
estimates store full

* First 50%
logit dvgunlaw gunhomicideslag1 citizenideologylag1 numregdvgunlawenactlag1 ///
    vawa1994 vawa1995 lautenbergamdt1996 Lautenbergamndt1997 ///
    legislature_election_year femleg innovation_index if sample_half == 1, vce(cluster state)
estimates store first

* Last 50%
logit dvgunlaw gunhomicideslag1 citizenideologylag1 numregdvgunlawenactlag1 ///
    vawa1994 vawa1995 lautenbergamdt1996 Lautenbergamndt1997 ///
    legislature_election_year femleg innovation_index if sample_half == 2, vce(cluster state)
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
display as text "Results saved to schiller_split_results.txt"