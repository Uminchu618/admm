# Example for Flexible AFT model using the SUPPORT2 inference dataset

library(survival)
library(splines)
source("FlexAFT.R")

Support2Dat <- read.csv("support2_inference.csv")

# FlexAFT expects one row per subject with no missing data. The supplied
# inference file contains six rows per id indexed by k, so use k == 0.
Support2Dat <- subset(Support2Dat, k == 0)

Var <- setdiff(names(Support2Dat), c("id", "k", "time", "event"))

sink("support2_flexible_AFT.txt")
support2_fit <- FlexAFT(
  Data = Support2Dat,
  Var = Var,
  NL = rep(0, length(Var)),
  TD = rep(1, length(Var)),
  nknot.NL = rep(NA, length(Var)),
  nknot.TD = rep(1, length(Var)),
  degree.NL = rep(NA, length(Var)),
  degree.TD = rep(2, length(Var)),
  nknot.bh = 2,
  degree.bh = 3,
  Time.Obs = "time",
  Delta = "event",
  knot_time = "eventtime",
  tol = 1e-4,
  ndivision = 50
)
sink()

coef_table <- data.frame(
  variable = names(support2_fit$coefficient),
  coefficient = as.numeric(support2_fit$coefficient),
  time_ratio = exp(as.numeric(support2_fit$coefficient)),
  row.names = NULL
)

td_table <- do.call(rbind, lapply(seq_along(support2_fit$Var), function(i) {
  data.frame(
    variable = support2_fit$Var[i],
    parameter = paste0("theta", seq_along(support2_fit$spline_coef_TD[[i]])),
    coefficient = as.numeric(support2_fit$spline_coef_TD[[i]]),
    row.names = NULL
  )
}))

bh_table <- data.frame(
  parameter = paste0("gamma", seq_along(support2_fit$spline_coef_bh)),
  coefficient = as.numeric(support2_fit$spline_coef_bh),
  row.names = NULL
)

write.csv(coef_table, "support2_flexaft_coefficients.csv", row.names = FALSE)
write.csv(td_table, "support2_flexaft_td_spline.csv", row.names = FALSE)
write.csv(bh_table, "support2_flexaft_baseline_hazard_spline.csv", row.names = FALSE)
save(support2_fit, coef_table, td_table, bh_table, file = "support2_flexaft.RData")

print(list(
  n = nrow(Support2Dat),
  events = sum(Support2Dat$event),
  variables = Var,
  logLikelihood = support2_fit$logLikelihood,
  df = support2_fit$df,
  runtime_seconds = support2_fit$runtime,
  td_spline = td_table,
  baseline_hazard_spline = bh_table
))
