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

use "data/boushey2016.dta", clear

***************************************************************
* Step 2. Keep variables of interest and drop missing
***************************************************************
keep state year styear dvadopt policycongruent gub_election elect2 hvd_4yr fedcrime ///
     leg_dem_per_2pty dem_governor insession propneighpol citidist squire_prof86 ///
     citi6008 crimespendpc crimespendpcsq violentthousand pctwhite stateincpercap ///
     logpop counter counter2 counter3
drop if missing(dvadopt, policycongruent, gub_election, elect2, hvd_4yr, fedcrime, ///
                leg_dem_per_2pty, dem_governor, insession, propneighpol, citidist, ///
                squire_prof86, citi6008, crimespendpc, crimespendpcsq, violentthousand, ///
                pctwhite, stateincpercap, logpop, counter, counter2, counter3)

***************************************************************
* Step 3. Sort by year and create split indicator (first 50% vs last 50%)
***************************************************************
* Find the midpoint year
summ year, meanonly
local midyear = floor((r(min) + r(max)) / 2)

* Create a variable that splits based on year
gen sample_half = cond(year <= `midyear', 1, 2)

***************************************************************
* Step 4. Run logistic regressions for each split
***************************************************************
* First half
logit dvadopt policycongruent gub_election elect2 hvd_4yr fedcrime ///
      leg_dem_per_2pty dem_governor insession propneighpol citidist squire_prof86 ///
      citi6008 crimespendpc crimespendpcsq violentthousand pctwhite stateincpercap ///
      logpop counter counter2 counter3 ///
      if sample_half == 1, vce(cluster styear)
estimates store first50

* Second half
logit dvadopt policycongruent gub_election elect2 hvd_4yr fedcrime ///
      leg_dem_per_2pty dem_governor insession propneighpol citidist squire_prof86 ///
      citi6008 crimespendpc crimespendpcsq violentthousand pctwhite stateincpercap ///
      logpop counter counter2 counter3 ///
      if sample_half == 2, vce(cluster styear)
estimates store last50

***************************************************************
* Step 5. Create Coefplot
***************************************************************
coefplot (first50, label("First 50%")) (last50, label("Last 50%")), ///
    drop(_cons state_*) nolabel xline(0, lpattern(dot)) ///
    bycoefs ///
    byopts(cols(5) xrescale) ///
    rename(policycongruent = "Policy Congruence" gub_election = "Elect1" ///
           elect2 = "Elect2" hvd_4yr = "Electoral Competition" fedcrime = "National Crime Salience" ///
           leg_dem_per_2pty = "Democratic Party Strength" dem_governor = "Democratic Governor" ///
           insession = "Legislative Session" propneighpol = "Neighbors" ///
           citidist = "Ideological Distance" squire_prof86 = "Legislative Professionalism" ///
           citi6008 = "Political Ideology" crimespendpc = "Crime Spending per Capita" ///
           crimespendpcsq = "Crime Spending (Squared)" violentthousand = "Violent Crime Rate" ///
           pctwhite = "Pct. Population White" stateincpercap = "Per Capita Income" ///
           logpop = "Logged Population" counter = "Time" counter2 = "Time Squared" counter3 = "Time Cubed") ///
    xtitle("Logit Coefficients") ///
    ylabel(none) ///
    legend(pos(6) rows(1))

graph export "ml_coefficient_split/figures/boushey2016/boushey_coefplot_split.png", replace width(2000)