# Flexible AFT Model
R code for implementing the method in article “Pang M, Platt R, Schuster T, Abrahamowicz M. Flexible Modeling of Time-dependent and Non-linear Covariate Effects in Accelerated Failure Time Model. Under review at *Statistical Methods in Medical Research* 2020".

## Content
### Description
A flexible extension of the accelerated failure time model that models:
- baseline hazard
- TD effects of continuous and categorical variables
- NL effects of continuous variables

The code has been written using R with the following version information:<br/>
- R version 3.6.3 (2020-02-29)<br/> 
- Platform x86_64-apple-darwin15.6.0 (64-bit)<br/> 
- Using R packages:<br/> 
  - survival version 3.1-12
  - splines version v3.6.2
  
#### Code to implement the flexible AFT model:
##### `FlexAFT.R`
This program includes all necessary functions to provide estimates of:
- TD effects
- NL effects
- Time ratio when TD effect is not present
- Hazard function and survival curve conditional on an arbitrary covariate pattern with their relevant TD and NL effects


The program is called by the program `Example_lungcancer.R`. 

#### Code to run the flexible AFT model:
##### `Example_lungcancer.R`
The program uses the lung cancer dataset that is available in R 'survival' package, and generates the results save in `lung_flexaft.RData`
 
For questions or comments about the code please contact Menglan Pang (menglan.pang at mail.mcgill.ca).
