# What is this page?
This page shows the example python code to apply statistical causal discovery (SCD) to development of adverse outcome pathway (AOP). The manuscript is inpreparation.
  
  
   
# Introduction  
Statistical causal-discovery (SCD) methods infer causal structure directly from observational and/or interventional data, by imposing tractable assumptions that restrict the search space of possible causal graphs. Traditional SCD methods, such as Peter and Clark (PC) and fast causal inference (FCI) algorithms, do not necessarily identify a fully directed graph and often leave several directions unresolved. However, some SCD methods can distinguish among causal structures within the same Markov-equivalence class, by making additional assumptions. For example, in the bivariate case, the linear non-Gaussian acyclic model (LiNGAM) distinguishes X → Y from Y → X when noise terms are non-Gaussian and independent (Shimizu et al., 2006; http://jmlr.org/papers/v7/shimizu06a.html).  
Our paper attempts to incorporate SCD into the adverse outcome pathway (AOP) framework for environmental risk assessment. 
   
   

# Files
1. Imputation_DirectLiNGAM_code.md  
An example python code for SCD analysis to develop AOP. Run this code in a directory containing "Missing_all.csv".  

2. Missing_all.csv  
The data used for SCD analysis with the above python code. The original data were taken from Moe et al. (2020; https://doi.org/10.1002/ieam.4348).  
