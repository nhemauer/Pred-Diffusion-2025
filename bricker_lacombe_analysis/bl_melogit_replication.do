// Estimate Bricker-Lacombe replication adding in the edge_sum variable.
// This estimates the monadic model on the full dataset.

capture log close
log using bl_melogit_replication_m1.log, replace text

use bricker_lacombe_neha_data, clear

  //// Estimate models from Table 2 - Pooled Monadic Event History Analysis

melogit adoption std_score initiative init_sigs ///
	std_pop std_citideology unified std_income std_legp_squire ///
	duration  durationsq durationcb i.year || policyno:

	estimates store m1
	estimates save bl_melogit_replication_m1, replace

melogit adoption std_score initiative init_sigs ///
	std_pop std_citideology unified std_income std_legp_squire ///
	duration  durationsq durationcb edge_sum i.year || policyno:

	estimates store m2
	estimates save bl_melogit_replication_m2, replace

  //// Make the table for our paper
  
label variable 	std_score 		"Similarity"
label variable 	initiative 		"Initiative Process"
label variable 	init_sigs 		"Signatures"
label variable 	std_pop  		"Population"
label variable 	std_citideology "Citizen Ideology"
label variable 	unified  		"Unified Control"
label variable 	std_income  	"Std. Income"
label variable 	std_legp_squire	"Leg. Professionalism"  
label variable 	duration   		"Duration"
label variable 	durationsq  	"Duration Squared"
label variable 	durationcb  	"Duration Cubed"
label variable 	edge_sum 		"gamma"
  
esttab m1 m2 using bl_melogit_replication.tex, replace style(tex) ///
	b(%9.3f) se noparen wide compress label varwidth(25) ///
	star(* 0.05) ///
	drop(*year*) ///
	eqlabels( , none) nodepvars mtitles("PEHA" "NEHA") nonumbers nonotes ///
	rename([adoption]_b[_cons] "Constant" var(_cons[policyno]):_cons "State~variance") 

clear
log close
exist, STATA
