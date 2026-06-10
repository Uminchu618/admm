# Plot fitted Flexible AFT results for the SUPPORT2 inference dataset

library(survival)
library(splines)
source("FlexAFT.R")

Support2Dat <- read.csv("support2_inference.csv")
Support2Dat <- subset(Support2Dat, k == 0)

load("support2_flexaft.RData")

Data <- Support2Dat
Time.Obs <- support2_fit$Time.Obs
Delta <- support2_fit$Delta
Var <- support2_fit$Var

time_grid <- seq(0.02, min(5.8, max(Data[, Time.Obs])), length.out = 120)
event_times <- Data[Data[, Delta] == 1, Time.Obs]
event_freq <- table(round(event_times, 3))
event_uniq <- as.numeric(names(event_freq))

binary_vars <- Var[sapply(Data[, Var], function(x) length(unique(x)) == 2)]
continuous_vars <- setdiff(Var, binary_vars)

cov.val <- sapply(Var, function(v) {
  if (v %in% binary_vars) 0 else median(Data[, v])
})

td_estimate <- function(var.name) {
  if (var.name %in% binary_vars) {
    TDest_bin(fit = support2_fit, Data = Data, var.name = var.name, time = time_grid)
  } else {
    TDest_con(
      fit = support2_fit,
      Data = Data,
      var.name = var.name,
      Q.high = 0.9,
      Q.low = 0.1,
      time = time_grid
    )
  }
}

add_event_rug <- function() {
  for (j in seq_along(event_uniq)) {
    try(rug(event_uniq[j], ticksize = min(0.06, 0.006 * event_freq[j])), silent = TRUE)
  }
}

plot_td <- function(var.name, show_exp = FALSE) {
  est <- td_estimate(var.name)
  y <- drop(est$est.TD)
  ylab <- expression(beta(t))
  if (show_exp) {
    y <- exp(y)
    ylab <- expression(e^beta(t))
  }
  ylim <- range(y, finite = TRUE)
  pad <- diff(ylim) * 0.08
  if (!is.finite(pad) || pad == 0) pad <- 0.1
  plot(
    est$time, y,
    type = "l",
    xlab = "Time",
    ylab = ylab,
    main = paste("TD effect:", var.name),
    ylim = ylim + c(-pad, pad),
    cex.axis = 0.8,
    cex.lab = 0.8,
    cex.main = 0.8
  )
  abline(h = if (show_exp) 1 else 0, col = "gray70", lty = 2)
  add_event_rug()
}

pdf("support2_Rplots.pdf", width = 10, height = 7.5)

par(mfrow = c(2, 3), mgp = c(2, 1, 0), mar = c(3.5, 3.5, 2, 1),
    oma = c(0.25, 0.25, 0.25, 0.25))

Hazard.est <- HazardEst(fit = support2_fit, time = time_grid, cov = cov.val, Data = Data)
plot(
  Hazard.est$time, Hazard.est$hazard,
  type = "l",
  xlab = "Time",
  ylab = "Hazard",
  main = "Hazard",
  cex.axis = 0.8,
  cex.lab = 0.8,
  cex.main = 0.8
)
add_event_rug()
mtext("(a)", side = 1, line = 2, at = min(time_grid), cex = 0.75)

Surv.est <- SurvEst(fit = support2_fit, time = time_grid, cov = cov.val, Data = Data)
plot(
  Surv.est$time, Surv.est$survival,
  type = "l",
  ylim = c(0, 1),
  xlab = "Time",
  ylab = "Survival",
  main = "Survival",
  cex.axis = 0.8,
  cex.lab = 0.8,
  cex.main = 0.8
)
add_event_rug()
mtext("(b)", side = 1, line = 2, at = min(time_grid), cex = 0.75)

summary_vars <- c("age", "sex_male", "ca_metastatic", "meanbp")
summary_vars <- summary_vars[summary_vars %in% Var]
for (i in seq_along(summary_vars)) {
  plot_td(summary_vars[i], show_exp = FALSE)
  mtext(paste0("(", letters[i + 2], ")"), side = 1, line = 2, at = min(time_grid), cex = 0.75)
}

for (start in seq(1, length(Var), by = 6)) {
  par(mfrow = c(2, 3), mgp = c(2, 1, 0), mar = c(3.5, 3.5, 2, 1),
      oma = c(0.25, 0.25, 0.25, 0.25))
  vars_page <- Var[start:min(start + 5, length(Var))]
  for (var.name in vars_page) {
    plot_td(var.name, show_exp = FALSE)
  }
}

dev.off()
