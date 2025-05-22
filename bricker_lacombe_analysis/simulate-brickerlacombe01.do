version 14
set more off
set matsize 2500
clear
  quietly log
  local logon = r(status)
  if "`logon'" == "on" { 
	log close 
	}
log using simulate-brickerlacombe01, text replace


/*	********************************************************************	*/
/* 	Author:		Frederick J. Boehmke										*/
/*	Date:		October 27, 2021											*/
/*  File:		simulate-brickerlacombe01.do								*/
/*	Purpose:	Code for generating predicted probabilities for initiative	*/
/*				and signature requirements for NEHA replications. Run 		*/
/*				bl_melogit_replication.do first to create saved estimates.	*/
/*	Input: 		bricker_lacombe_neha_data.dta								*/
/*				bl_melogit_replication_m1.ster								*/
/*				bl_melogit_replication_m2.ster								*/
/*	Output: 	simulate-brickerlacombe01.log								*/
/*				simulate-brickerlacombe01.dta								*/
/*				simulate-brickerlacombe01-XXX.png							*/
/*	********************************************************************	*/


clear
save simulate-brickerlacombe01, replace emptyok

		/* Open estimates for the replication model */

use ./replication_data/bricker_lacombe_neha_data, clear
   
 
	/*******************************************************************/
	/* Run margins and save the results into a data set to plot later. */
	/*******************************************************************/
	
		/* Do first differences for the covariates of interest. */
		/* Margins reads left to right, so okay to specify variables twice. */
		/* Save the predicted probabilities into a data set for differencing later. */
		/* Account for the difference in the distribution of major topics, time, */
		/* etc. by averaging over them rather than fixing values. */

forvalues i = 1/2 {

	if `i' == 1 {
		local edge_sum = ""
		}
	else if `i' == 2 {
		local edge_sum = "edge_sum"
		}

		/* Start with the continuous variables. */
		/* This is MIP when complex = 0. */
		/* This should average over time and majortopic. */

	foreach var of varlist std_pop std_citideology std_income std_legp_squire `edge_sum' {

		preserve

		estimates use bl_melogit_replication_m`i'

		estimates esample: adoption std_score initiative init_sigs ///
			std_pop std_citideology unified std_income std_legp_squire ///
			duration durationsq durationcb year `edge_sum',  stringvars(policyno) replace
			
		margins, predict(mu) ///
			at((asobserved) std_score initiative init_sigs ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year (p16) `var') /// 
			at((asobserved) std_score initiative init_sigs ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year (p84) `var') post
					
		matrix M = r(table)
		matrix A = r(at)

		matrix M = M'

		svmat M, names(col) 
				
		generat at_model = `i' 
		generat at_var = "`var'"
		generat at_val = "low" in 1
		replace at_val = "hi" in 2
		
		matrix L = A[1,"`var'"]
		matrix H = A[2,"`var'"]

		nlcom _b[2._at] - _b[1._at]	
			
		generat fd = el(r(b),1,1) in 1
		generat fd_se = sqrt(el(r(V),1,1)) in 1
				
		generat at_val_num = el(L,1,1) in 1
		replace at_val_num = el(H,1,1) in 2 
		
		keep b-at_val_num
		drop if missing(b)
		
		append using simulate-brickerlacombe01
		save simulate-brickerlacombe01, replace
		
		restore
	
		}
		
		/* This is for binary variables besides the initiative. */

	foreach var of varlist unified {
	
		preserve

		estimates use bl_melogit_replication_m`i'

		estimates esample: adoption std_score initiative init_sigs ///
			std_pop std_citideology unified std_income std_legp_squire ///
			duration durationsq durationcb year `edge_sum',  stringvars(policyno) replace
			
		margins, predict(mu) ///
			at((asobserved) std_score initiative init_sigs ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=0) /// 
			at((asobserved) std_score initiative init_sigs ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=1) post 
					
		matrix M = r(table)
		matrix A = r(at)

		matrix M = M'

		svmat M, names(col) 

		generat at_model = `i' 
		generat at_var = "`var'"
		generat at_val = "low" in 1
		replace at_val = "hi" in 2
		
		matrix L = A[1,"`var'"]
		matrix H = A[2,"`var'"]

		generat at_val_num = el(L,1,1) in 1
		replace at_val_num = el(H,1,1) in 2
		
		nlcom _b[2._at] - _b[1._at], iterate(1000)		
			
		generat fd = el(r(b),1,1) in 1
		generat fd_se = sqrt(el(r(V),1,1)) in 1
				
		keep b-at_val_num
		drop if missing(b)
		
		append using simulate-brickerlacombe01
		save simulate-brickerlacombe01, replace
		
		restore
	
		}
				
		/* This is for the initiative when signatures are zero. */

	foreach var of varlist initiative {
	
		preserve

		estimates use bl_melogit_replication_m`i'

		estimates esample: adoption std_score initiative init_sigs ///
			std_pop std_citideology unified std_income std_legp_squire ///
			duration durationsq durationcb year `edge_sum',  stringvars(policyno) replace
			
		margins, predict(mu) ///
			at((asobserved) std_score initiative init_sigs=0 ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=0) /// 
			at((asobserved) std_score initiative init_sigs=0 ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=1) post 
									
		matrix M = r(table)
		matrix A = r(at)

		matrix M = M'

		svmat M, names(col) 

		generat at_model = `i' 
		generat at_var = "`var'"
		generat at_val = "low" in 1
		replace at_val = "hi" in 2
					
		matrix L = A[1,"`var'"]
		matrix H = A[2,"`var'"]
	
		nlcom _b[2._at] - _b[1._at], iterate(1000)		
			
		generat fd = el(r(b),1,1) in 1
		generat fd_se = sqrt(el(r(V),1,1)) in 1
		
		generat at_val_num = el(L,1,1) in 1
		replace at_val_num = el(H,1,1) in 2
		
		keep b-at_val_num
		drop if missing(b)
		
		append using simulate-brickerlacombe01
		save simulate-brickerlacombe01, replace
		
		restore
	
		}
				
		/* This is for initiative presence and signatures. */
		
	_pctile init_sigs if initiative==1, percentile(86)
	
	local sigs_p86 = r(r1)

	foreach var of varlist init_sigs {
	
		preserve

		estimates use bl_melogit_replication_m`i'

		estimates esample: adoption std_score initiative init_sigs ///
			std_pop std_citideology unified std_income std_legp_squire ///
			duration durationsq durationcb year `edge_sum',  stringvars(policyno) replace
			
		margins, predict(mu) ///
			at((asobserved) std_score initiative=0 init_sigs=0 ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=0) /// 
			at((asobserved) std_score initiative=1 init_sigs=0 ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year `var'=`sigs_p86') post 
										
		matrix M = r(table)
		matrix A = r(at)

		matrix M = M'

		svmat M, names(col) 

		generat at_model = `i' 
		generat at_var = "`var'"
		generat at_val = "low" in 1
		replace at_val = "hi" in 2
		
		matrix L = A[1,"`var'"]
		matrix H = A[2,"`var'"]

		nlcom _b[2._at] - _b[1._at], iterate(1000)		
			
		generat fd = el(r(b),1,1) in 1
		generat fd_se = sqrt(el(r(V),1,1)) in 1
		
		generat at_val_num = el(L,1,1) in 1
		replace at_val_num = el(H,1,1) in 2
		
		keep b-at_val_num
		drop if missing(b)
		
		append using simulate-brickerlacombe01
		save simulate-brickerlacombe01, replace
		
		restore
	
		}
		
	}
	
			/*****************************************************************/
			/* Do predicted probabilities varying initiative and signatures. */
			/*****************************************************************/

	
clear

save simulate-brickerlacombe01-initsigs, replace emptyok

use bricker_lacombe_neha_data, clear
   
forvalues i = 1/2 {

	preserve

	if `i' == 1 {
		local edge_sum = ""
		}
	else if `i' == 2 {
		local edge_sum = "edge_sum"
		}

	estimates use bl_melogit_replication_m`i'

	estimates esample: adoption std_score initiative init_sigs ///
		std_pop std_citideology unified std_income std_legp_squire ///
		duration durationsq durationcb year `edge_sum',  stringvars(policyno) replace
		
		margins, predict(mu) ///
			at((asobserved) std_score initiative=0 init_sigs=0 ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum') /// 
			at((asobserved) std_score initiative=1 init_sigs=(0/15) ///
			  std_pop std_citideology unified std_income std_legp_squire ///
			  duration durationsq durationcb `edge_sum' year) post 
									
		matrix M = r(table)
		matrix A = r(at)

		clear
		
		matrix M = M'
		
		matrix L = A[.,"initiative"]
		matrix H = A[.,"init_sigs"]
			
		svmat M, names(col) 
		svmat L, names(col) 
		svmat H, names(col) 

		generat fd = .
		generat fd_se = .

		forvalues j = 2/16 {
		
			nlcom _b[`j'._at] - _b[1._at]
				
			replace fd = el(r(b),1,1) in `j'
			replace fd_se = sqrt(el(r(V),1,1)) in `j'
			
			}
		
		generat at_model = `i' 
					
		append using simulate-brickerlacombe01-initsigs
		save simulate-brickerlacombe01-initsigs, replace
		
		restore
		
	}
	
	
use simulate-brickerlacombe01-initsigs, clear

	generat fd_hi = fd + 1.96*fd_se			
	generat fd_lo = fd - 1.96*fd_se			

	label define models 1 "PEHA" 2 "NEHA"

	label values at_model models 
	
		/* Graph the predicted probabilities. */
	
twoway rarea ll ul init_sigs if initiative==1, by(at_model, legend(off) note("") iscale(*1.3)) scheme(s1mono) ///
	color(gs13) ///
  || line b init_sigs if initiative==1, by(at_model) ///
	lcolor(black) lwidth(medthick) ///
  || rcap ll ul init_sigs if initiative==0, by(at_model) ///
	lcolor(gs7) lwidth(medthick) ///
  || scatter b init_sigs if initiative==0, by(at_model) ///
	mcolor(black) msymbol(O) msize(large) ///
	xtitle(Signatures, size(*1.5)) ///
	ytitle(Probability of Adoption, size(*1.5)) ///
	text(0.068 0.5 "No initiative", place(e)) ///
	text(0.09 5 "Initiative and signatures", place(e)) ///
	ysize(3) xsize(5)

	graph export simulate-brickerlacombe01-probs.png, replace width(3000)
	
		/* Graph the FDs. */
	
twoway rarea fd_lo fd_hi init_sigs if initiative==1, by(at_model, legend(off) note("") iscale(*1.3)) scheme(s1mono) ///
	color(gs13) ///
  || line fd init_sigs if initiative==1, by(at_model) ///
	lcolor(black) lwidth(medthick) ///
	xtitle(Signatures, size(*1.5)) ///
	ytitle(Change in Probability of Adoption, size(*1.5)) ///
	ysize(3) xsize(5)

	graph export simulate-brickerlacombe01-fds.png, replace width(3000)
	
	
	
log close
clear
exit, STATA
