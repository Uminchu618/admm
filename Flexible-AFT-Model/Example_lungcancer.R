# Example for Flexible AFT model using the lung cancer dataset that is available in R 'survival' package

library(survival)
library(splines)
# setwd("…")
source("FlexAFT.R")

LungDat <- lung

LungDat$time.yr <- LungDat$time/365
LungDat$event <- LungDat$status-1
LungDat$sex <- LungDat$sex-1
LungDat$wt.loss[is.na(LungDat$wt.loss)] <- mean(LungDat$wt.loss, na.rm=TRUE)


##### Fit the flexible AFT model #####

## Input for FlexAFT function:
# Data: dataset (no missing data allowed; categorial variables have to be coded as dummy variables)
# Var: names of the independent variables to be included in the flexible AFT model
# NL: indicate for each variable if the NL effect is modeled (0/1)
#     Can be 1 only for continuous variables
# TD: indicate for each variable if the TD effect is modeled (0/1)
# nknot.NL: number of interior knots of the splines for NL effects. Put NA when no NL effect for a variable
# nknot.TD: number of interior knots of the splines for TD effects. Put NA when no TD effect for a variable
# degree.NL: degree of the splines for NL effects. Put NA when no NL effect for a variable
# degree.TD: degree of the splines for TD effects. Put NA when no TD effect for a variable
# nknot.bh: number of interior knots of the splines for modeling the baseline hazard
# degree.bh: degree of the splines for modeling the baseline hazard
# Time.Obs: name of the observed time variable
#     (NOTE: converting the time variable to years or months, instead of days, can help reduce potentially long computation times)
# Delta: name of the event indicator
# knot_time: "eventtime" or "alltime" specifying whether the knots are placed based on the quantiles of the 
#     uncensored event times or the overall observed (event or censoring) times. The default is "alltime".
# tol: convergence criterion for log-likelihood between two consecutive iterations. The default is 1^-5. A lager value, i.e.,relaxed convergence critria will
#     reduce the computation time and may potentially result in less accurate estimates. Users are encouraged to first test the program by specifiying a larger convergence critria, then 
#     gradually decrease the value and compare the resulting log-likelihood and the estimates.
# ndivision: the number of intervals defining the granularity of the numerical computation of integrals.
#     The range of the integral is divided by ndivision small intervals.
#     The larger the number is, the longer the estimation takes with better accuracy.
#     The default value is 100.  
#     When deciding a customized value for ndivision for the trade-off between estimation accuracy and computation time,
#     one may take into account the maximum of Time.Obs and how quickly the underlying hazard function is expected to change.
#     For example, the the maximum value of the follow-up time is 3 years, then the numerical integration approximation is
#     based on small intervals of approximately (3*365/100) 10 days, which is considered as very good granularity.
#     With shorter/longer follow-up time, smaller/larger values for ndivision can be specificed.
#     In general, a larger value for nvidision is preferred for estimating a
#     an underlying hazard function with rapid change, whereas a smaller value
#     may be sufficient if the underlying hazard is expected to be very smooth.
#     Users are encouraged to try a few values for this option and compare the
#     resulting log-likelihood and the estimates of the hazard function.

sink("lung_flexible_AFT.txt")  # Save details of each iteration
lung_fit <- FlexAFT(Data=LungDat, Var=c("sex","age","wt.loss"),
                    NL=c(0,1,1), TD=c(1,0,1),
                    nknot.NL=c(NA,1,1), nknot.TD=c(1,NA,1),
                    degree.NL=c(NA,2,2), degree.TD=c(2,NA,2),
                    nknot.bh=2, degree.bh=3,
                    Time.Obs="time.yr", Delta="event",
                    knot_time="eventtime", tol=1e-5)
sink()
save(lung_fit, file="lung_flexaft.RData")


#### Estimation of the hazard function #####

# Covariate pattern for which the hazard function is estimated
cov.val <- rep(NA,3)
cov.val[1] <- 0
cov.val[2] <- mean(LungDat$age)
cov.val[3] <- mean(LungDat$wt.loss)


par(mfrow=c(2,3), mgp=c(2,1,0), mar=c(3.5,3.5,2,1),
    oma=c(0.25,0.25,0.25,0.25))

## Input for HazardEst function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# time: a vector of time values at which the hazard function is estimated.
#     It is necessary to restrict the values within the range of the observed time variable in the data to avoid
#     error and data extrapolation.
# cov: the specific covariate pattern for which the hazard function is estimated
#     Note: Warnings may occur for some covariate patterns that yield exp(beta(t)g(x)t beyond the values in the empirical dataset. #     This is an indication that the estimates are obtained by extrapolation for the provided covariate pattern at some of the
#     time points among the time vector, most likely at large t. 
# Data: dataset that was used to fit the flexible AFT model
Hazard.est <- HazardEst(fit=lung_fit, time=seq(0.02,2.8,0.1), cov=cov.val, Data=LungDat)

plot(0,0,type="n",xlim=c(0,2.8),ylim=c(0,2),xlab="Time (day)",ylab="Hazard",cex.main=0.75,main="Hazard")
lines(Hazard.est$time,Hazard.est$hazard)
Data <- LungDat
Time.Obs <- "time.yr"
Delta <- "event"

# Add rug plot in the bottom of the graph for the distribution of the event time distribution
Time.obs <- Data[Data[,Delta]==1,Time.Obs]
Time.obs.freq <- table(Time.obs)
Time.obs.uniq <- as.numeric(names(Time.obs.freq))
for (j in 1:length(Time.obs.uniq)){
  try(rug(Time.obs.uniq[j],ticksize=0.03*Time.obs.freq[j]),silent=TRUE)
}
mtext('(a)', side=1, line=2, at=0,cex=0.75)


##### Estimation of TD effect for a binary variable #####

plot(0,0,type="n",xlim=c(0,2.8),ylim=c(0.6,0.66),ylab=expression(e^beta(t)),xlab="Time(Day)",cex.axis=0.8,cex.lab=0.8,cex.main=0.75,main="TD effect for Sex")

## Input for TDest_bin function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# var.name: name of one variable whose TD effect is to be estimated
# time: a vector of time values at which the TD effect is to be estimated

TD_effect <- TDest_bin(fit=lung_fit, Data=LungDat, var.name="sex", time=seq(0.02,2.8,0.1))

lines(TD_effect$time,exp(TD_effect$est.TD),lty=1,lwd=1)


for (j in 1:length(Time.obs.uniq)){
  try(rug(Time.obs.uniq[j],ticksize=0.03*Time.obs.freq[j]),silent=TRUE)
}
mtext('(b)', side=1, line=2, at=0,cex=0.75)


##### Estimation of a NL effect ####

plot(0,0,type="n",xlim=c(39,82),ylim=c(-0.8,1.5),xlab="Age",ylab=expression(g(X)),cex.axis=0.8,cex.lab=0.8,cex.main=0.75,main="NL effect for Age")

## Input for NLest function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# var.name: name of one variable whose TD effect is to be estimated
# ref.val: the reference value to which the NL effect is relatively estimated
# Q.high, Q.low: quantiles of the variable where the risk at Q.high is higher than Q.low.
#     These arguments are to ensure the estimate is in the direction making sense. 
#     Therefore, any quantile values for which the risk is known to be higher (Q.high) than the second value (Q.low)
#     are appropriate.
#     For example, setting Q.high=0.9 and Q.low=0.1 for age, by using substantive knowledge, means that age at 90th quantile
#     has higher risk of mortality than age at 10th quantile.
# var.val: a vector of variable values at which the NL effect is to be estimated



NL_est <- NLest(fit=lung_fit, Data=LungDat, var.name="age",
                ref.val=62, Q.high=0.9, Q.low=0.1, var.val=seq(39,82,0.1))

lines(NL_est$X,NL_est$est.NL)

# Add rug plot in the bottom of the graph for the distribution of the independent variable
X.freq <- table(LungDat[,"age"])
X.uniq <- as.numeric(names(X.freq))
for (j in 1:length(X.uniq)){
  try(rug(X.uniq[j],ticksize=X.freq[j]/80),silent=TRUE)
}
mtext('(c)', side=1, line=2, at=40,cex=0.75)


##### Estimation TD effect for a continuous variable ######

plot(0,0,type="n",xlim=c(0,2.8),ylim=c(-1.1,8),ylab=expression(beta(t)),xlab="Time (Day)",cex.axis=0.8,cex.lab=0.8,cex.main=0.75,main="TD effect for weight loss")

## Input for TDest_con function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# var.name: name of one variable whose TD effect is to be estimated
# Q.high, Q.low: quantiles of the variable where the risk at Q.high is higher than Q.low
#     These arguments are to ensure the estimate is in the direction making sense. 
#     Therefore, any quantile values for which the risk is known to be higher (Q.high) than the second value (Q.low)
#     are appropriate.
#     For example, setting Q.high=0.9 and Q.low=0.1 for age, by using substantive knowledge, means that age at 90th quantile
#     has higher risk of mortality than age at 10th quantile.
# time: a vector of time values at which the TD effect is to be estimated

TD_est <- TDest_con(fit=lung_fit, Data=LungDat, var.name="wt.loss",
                    Q.high=0.9, Q.low=0.1, time=seq(0.02,2.8,0.1))

lines(TD_est$time,TD_est$est.TD)

# Add rug plot in the bottom of the graph for the distribution of the event time distribution
for (j in 1:length(Time.obs.uniq)){
  try(rug(Time.obs.uniq[j],ticksize=0.03*Time.obs.freq[j]),silent=TRUE)
}
mtext('(d)', side=1, line=2, at=0,cex=0.75)

plot(0,0,type="n",xlim=c(-24,68),ylim=c(-0.8,0.5),xlab="Weigth Loss",ylab=expression(g(X)),cex.axis=0.8,cex.lab=0.8,cex.main=0.75,main="NL effect for Weight Loss")
NL_est <- NLest(fit=lung_fit, Data=LungDat, var.name="wt.loss",
                ref.val=10, Q.high=0.9, Q.low=0.1, var.val=seq(-24,68,0.1))

lines(NL_est$X,NL_est$est.NL)

# Add rug plot in the bottom of the graph for the distribution of the independent variable
X.freq <- table(LungDat[,"wt.loss"])
X.uniq <- as.numeric(names(X.freq))
for (j in 1:length(X.uniq)){
  try(rug(X.uniq[j],ticksize=X.freq[j]/100),silent=TRUE)
  
}
mtext('(e)', side=1, line=2, at=-20,cex=0.75)


##### Estimation of survival curves #####

## Input for SurvEst function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# time: a vector of time values at which the survival curve is estimated
# cov: the specific covariate pattern for which survival curve is estimated
# Data: dataset that was used to fit the flexible AFT model

Surv.est <- SurvEst(fit=lung_fit, time=seq(0.02,2.8,0.1), cov=cov.val, Data=LungDat)

plot(0,0,type="n",xlim=c(0,2.8),ylim=c(0,1),xlab="Time (day)",ylab="Survival",cex.main=0.75,main="Survival")
lines(Surv.est$time,Surv.est$survival)

# Add rug plot in the bottom of the graph for the distribution of the event time distribution
for (j in 1:length(Time.obs.uniq)){
  try(rug(Time.obs.uniq[j],ticksize=0.03*Time.obs.freq[j]),silent=TRUE)
}
mtext('(f)', side=1, line=2, at=0,cex=0.75)


##### Estimation of TD time ratios #####

# Covariate patterns (reference and index) for the time ratio comparison
X.ref <- rep(NA,3)
X.ref[1] <- 0
X.ref[2] <- 62
X.ref[3] <- 10

X.index <- X.ref
X.index[1] <- 1

# A vector of quantiles of event times at which the TD time ratio is estimated
pcttime <- c(0.95,0.9,0.85,0.8,0.75,0.7,0.5,0.4)

## Input for TDTRest function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# time: a vector of time values at which the TD time ratio is estimated
# cov0: reference covariate pattern for the time ratio comparison
# cov1: index covariate pattern for the time ratio comparison
# pct: a vector of quantiles of event times (0<pct<1) at which the TD time ratio is estimated

## Output from TDTRest function:
# $t.ref: the times when subjects with the reference covariate pattern attain each of the quantiles
# $t.index: the times when subjects with the index covariate pattern attain each of the quantiles
# $pct: the same vector of quantiles of event times (0<pct<1) at which the TD time ratio is estimated
# $TD_tr: the time-dependent time ratio estimated at each quantile of times comparing the index covariate pattern to
#     the reference covariate pattern

## Note: pct can be also viewed as the target proportion of subjects who have remained event free. 
# In the case that either the reference or index group does not achieve a specific pct value by the end of follow-up,
# the time-dependent time ratio cannot be estimated for this pct value. Therefore, the TDTRest would return NA for TD_tr 
# for this pct value.

sex.tr <- TDTRest(fit=lung_fit, Data=LungDat, time=seq(0.02,2.8,0.001), cov0=X.ref, cov1=X.index, pct=pcttime)

X.index <- X.ref
X.index[2] <- 70
age.tr <- TDTRest(fit=lung_fit, Data=LungDat, time=seq(0.02,2.8,0.001), cov0=X.ref, cov1=X.index, pct=pcttime)

X.index <- X.ref
X.index[3] <- 20
wt.loss.tr <- TDTRest(fit=lung_fit, Data=LungDat, time=seq(0.02,2.8,0.001), cov0=X.ref, cov1=X.index, pct=pcttime)


##### Calculate the linear predictor for a new dataset when applying the fitted flexible AFT model #####

## Input for pred.lp function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# time: a specific time point at which the linear preditor is computed
# newdata: a new dataset that contains the variables for which the linear predictor is computed.
#     In the example below, the original dataset is used for demonstration purposes.

predlp <- pred.lp(fit=lung_fit, Data=LungDat, time=2, newdata=LungDat)


##### Calculate the log-likelihood for a new dataset when applying the fitted flexible AFT model #####

## Input for AFTLoglike function:
# fit: object obtained after fitting the flexible AFT model using the "FlexAFT" function
# Data: dataset that was used to fit the flexible AFT model
# newdata: a new dataset that contains the variables for which the loglikelihood is computed
#     In the example below, the original dataset is used for demonstration purposes.
loglik <- AFTLoglike(fit=lung_fit, Data=LungDat, newdata=LungDat)


#### Calculate the AIC value ####

AIC <- 2*(lung_fit$df-loglik$LogLik)
