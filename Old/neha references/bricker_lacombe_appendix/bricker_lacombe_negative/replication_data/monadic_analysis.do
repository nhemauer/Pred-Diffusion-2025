/// monadic on full dataset
  use monadic-analysis_largen.dta, clear

  
  //// Creating Table 2- Pooled Monadic Event History Analysis
 melogit adoption std_score initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year || policyno:
 estimates store m1
 

 melogit adoption std_nbrs initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year || policyno:
 estimates store m2
 
  melogit adoption std_score std_nbrs initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year || policyno:
 estimates store m3
 
 
esttab m1 m2 m3 using models/monadic_1990_2016.tex, b(3) order(std_sim std_nbrs) compress drop(*.year) se label replace ///
	cells(b(fmt(4) star) se(par)) starlevels(* .1 ** .05 *** .01) /// 
	rename(_cons constant)  mlabels(none) varwidth(30) collabels(, none) eqlabels(, none)

/// now lets see if on more recent adoptions it matters more/less
melogit adoption std_sim initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year if first_year>=2010 || policyno:
 estimates store m1

 melogit adoption std_nbrs initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year if first_year>=2010 || policyno:
 estimates store m2
 
  melogit adoption std_sim std_nbrs initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year if first_year>=2010 || policyno:
 estimates store m3

 esttab m1 m2 m3, b(3) order(std_sim std_nbrs) compress drop(*.year) se label replace ///
	cells(b(fmt(4) star) se(par)) starlevels(* .1 ** .05 *** .01) /// 
	rename(_cons constant)  mlabels(none) varwidth(30) collabels(, none) eqlabels(, none)

	
 esttab m1 m2 m3 using models/monadic_2010_2016.tex, b(3) stats(aic bic) order(std_sim std_nbrs) compress drop(*.year) se label replace ///
	cells(b(fmt(4) star) se(par)) starlevels(* .1 ** .05 *** .01) /// 
	rename(_cons constant)  mlabels(none) varwidth(30) collabels(, none) eqlabels(, none)

	
/// predicted prob, non standardized sim measure
melogit adoption score_lag initiative init_sigs ///
 std_pop std_citideology unified std_income std_legp_squire ///
 duration  durationsq durationcb i.year || policyno:
 estimates store m1
 
 margins, at(score_lag=(0(.03).21)) predict(mu fixed)
marginsplot, name(sim_lag, replace) xtitle("Sum of Similarity Scores from Previous Adopters") ///
ytitle("Probability of Adoption") xscale(range(0 .21)) yscale(range(0 .25)) ///
 title("Probability of a Adoption") scheme(s2mono) recast(line) recastci(rarea) ///
 addplot(hist score_lag if score_lag<=.21, percent yaxis(2) ///
 xscale(range(0 .21)) yscale(alt axis(2) range(0 100)) legend(off)) 
graph export "monadic_prob.pdf", as(pdf) replace
graph export "monadic_prob.png", as(png) replace
graph save monadic_prob, replace

