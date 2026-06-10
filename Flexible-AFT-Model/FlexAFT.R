###
### Program for flexible AFT model in time-to-event analysis
###
# Author: Menglan Pang
# Last update: May 24, 2021

# Reference(under review at SMMR):
# Pang M, Platt RW, Schuster T, Abrahamowicz M. 
# Flexible modeling of time-dependent and non-linear covariate effects
# in Accelerated Failure Time model(2020).


### Functions in this program:

# For models with NT and/or NL effects:
# FlexAFT 
# AFT.logLik.alpha
# AFT.logLik.alpha.der
# AFT.logLik.beta
# AFT.logLik.beta.der
# AFT.logLik.gamma
# AFT.logLik.gamma.der
# integrand
# integrat
# TDest_bin
# TDest_con
# NLest
# HazardEst
# S.integrand
# SurvEst
# TDTRest
# pred.lp
# AFTLoglike

get_args_for <- function(fun, env = parent.frame(), inherits = FALSE, ..., dots) {
  potential <- names(formals(fun))
  
  if ("..." %in% potential) {
    if (missing(dots)) {
      # return everything from parent frame
      return(as.list(env))
    }
    else if (!is.list(dots)) {
      stop("If provided, 'dots' should be a list.")
    }
    
    potential <- setdiff(potential, "...")
  }
  
  # get all formal arguments that can be found in parent frame
  args <- mget(potential, env, ..., ifnotfound = list(NULL), inherits = inherits)
  # remove not found
  args <- args[sapply(args, Negate(is.null))]
  # return found args and dots
  c(args, dots)
}

library(survival)
library(splines)
#################################### Main Function ###########################################
###Initial Values:
#by default
#gamma_init=rate0 from exponential AFT
#beta_init=beta from exponential AFT asssuming constant time ratio
#alpha_init from linear regression: X~bs(X) assuming linearity;

###Bst are the spline basis functions for time; bs(t)
###Ax are the spline basis functions for X; bs(X)
###The product term corresponding to TD and NL specification
###TD=1 and NL=1, Bs(t)*g(x)=Bst*Ax
###TD=1 and NL=0, Bs(t)=Bst<-Bst*X; g(x)=Ax<-1
###TD=0 and NL=1, Bs(t)=Bst<-1; g(x)=Ax
###TD=0 and NL=0; Bs(t)=Bst<-X; g(x)=Ax<-1

FlexAFT<-function(Data,Var,NL,TD,nknot.NL,nknot.TD,degree.NL,degree.TD,nknot.bh,degree.bh,Time.Obs,Delta,knot_time="alltime",tol=1e-5,ndivision=100,init.beta="exp",init.gamma="exp",init.alpha="linear"){
  
  ##get the initial value of gamma from an exponential AFT model (constant baseline hazard)
  if (init.gamma=="exp"){
    exp_formula<-paste0("Surv(",Time.Obs,",",Delta,")~",paste0(Var,collapse = "+"))
    exponential<-survreg(formula(exp_formula),dist="exponential",data=Data)
    log_rate0<--exponential$coef[1]
  } else{
    log_rate0<-init.gamma 
  }
  #Observed time
  Tt<-as.matrix(Data[,Time.Obs])
  
  #event indicator
  delta<-Data[,Delta]
  
  #number of covariates included in the model
  nvar<-length(Var)
  
  #number of events
  num.event<-sum(delta)
  
  #number of parameter for modeling each covariate X 
  df.X<-nknot.NL+degree.NL+1
  
  #number of parameter for modeling beta(t) for each covariate
  df.bt<-nknot.TD+degree.TD+1
  df.bt[NL==0 & TD==0]<-1
  
  bh.quantile<-seq(1,nknot.bh)/(nknot.bh+1)
  
  alpha.knots<-vector("list",length=nvar)
  beta.knots<-vector("list",length=nvar)
  Ax<-vector("list",length=nvar)
  Bst<-vector("list",length=nvar)
  
  ##Create the spline basis functions for X and for beta(t);
  #depending on the specification of NL and TD effect
  
  ###Bst are the spline basis functions for time; bs(t)
  ###Ax are the spline basis functions for X; bs(X)
  ###TD=1 and NL=1, Bs(t)*g(x)=Bst*Ax; both alpha and beta are being estimted; TD and NL effect for X
  ###TD=1 and NL=0, Bs(t)=Bst<-Bst*X; g(x)=Ax<-1; only beta is being estimated; TD effect for X
  ###TD=0 and NL=1, Bs(t)=Bst<-1; g(x)=Ax; only alpha is being estimated; NL effect for X
  ###TD=0 and NL=0; Bs(t)=Bst<-X; g(x)=Ax<-1; only beta is being estimated; constant effect for X
  
  
  for (i in 1:nvar){
    if (NL[i]==1){
      nl.quantile<-seq(1,nknot.NL[i])/(nknot.NL[i]+1)
      alpha.knots[[i]]<-quantile(Data[,Var[i]],probs=nl.quantile)
      Ax[[i]]<-bs(Data[,Var[i]],knots=alpha.knots[[i]],degree=degree.NL[i],intercept=TRUE)
    } else {
      alpha.knots[[i]]<-NA;Ax[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      td.quantile<-seq(1,nknot.TD[i])/(nknot.TD[i]+1)
      if (knot_time=="alltime"){
        beta.knots[[i]]<-quantile(Tt,probs=td.quantile)
      } else if (knot_time=="eventtime"){
        beta.knots[[i]]<-quantile(Tt[delta==1],probs=td.quantile)
      }
      Bst[[i]]<-bs(Tt,knots=beta.knots[[i]],degree=degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      td.quantile<-seq(1,nknot.TD[i])/(nknot.TD[i]+1)
      if (knot_time=="alltime"){
        beta.knots[[i]]<-quantile(Tt,probs=td.quantile)
      } else if (knot_time=="eventtime"){
        beta.knots[[i]]<-quantile(Tt[delta==1],probs=td.quantile)
      }
      Bst[[i]]<-bs(Tt,knots=beta.knots[[i]],degree=degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*Data[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      beta.knots[[i]]<-NA;Bst[[i]]<-NA
    } else if (NL[i]==0 & TD[i]==0) {
      beta.knots[[i]]<-NA;Bst[[i]]<-matrix(Data[,Var[i]],ncol=1)}
  }
  
  if (knot_time=="alltime"){
    bh.knots<-quantile(Tt,probs=bh.quantile)
  } else if (knot_time=="eventtime"){
    bh.knots<-quantile(Tt[delta==1],probs=bh.quantile)
  } 
  #calculate df for gamma
  df.gamma<-nknot.bh+degree.bh+1
  
  #assign initial value to gamma
  gammaVec<-rep(log_rate0,df.gamma)
  names(gammaVec)<-NULL
  
  df.X.pos<-df.X
  df.X.pos[is.na(df.X)]<-0
  df.X.pos<-cumsum(df.X.pos)
  
  df.bt.pos<-df.bt
  df.bt.pos[is.na(df.bt)]<-0
  df.bt.pos<-cumsum(df.bt.pos)
  
  pos.alpha<-vector("list",nvar)
  pos.beta<-vector("list",nvar)
  #calculate df for alpha corresponding to each covariate X 
  pos<-0
  alphaVec<-rep(0,0)
  for (i in 1:nvar){
    if (NL[i]==1){
      pos<-pos+1
      pos.alpha[[i]]<-pos:df.X.pos[i]
      pos<-df.X.pos[i]
      #assign initial value to alpha
      if (init.alpha=="linear"){
        alphaVec<-c(alphaVec,lm(Data[,Var[i]]~-1+Ax[[i]])$coef)
      } 
    } else {pos.alpha[[i]]<-NA}
  }  
  #total df for NL effects
  df.alpha<-pos
  names(alphaVec)<-NULL
  
  if (init.alpha!="linear"){
    alphaVec<-rep(init.alpha,df.alpha)
  }
  
  
  #calculate df for beta corresponding to each covariate X 
  pos<-0
  betaVec<-rep(0,0)
  for (i in 1:nvar){
    if (TD[i]==1 | (TD[i]==0 & NL[i]==0)){
      pos<-pos+1
      pos.beta[[i]]<-pos:df.bt.pos[i]
      pos<-df.bt.pos[i]
      #assign initial value to beta
      if (init.beta=="exp"){
        betaVec<-c(betaVec,rep(-exponential$coef[Var[i]],df.bt[i]))
      } 
    } else {pos.beta[[i]]<-NA}
  } 
  #total df for TD effects
  df.beta<-pos
  names(betaVec)<-NULL
  
  if (init.beta!="exp"){
    betaVec<-rep(init.beta,df.beta)
  }
  
  starttime<-proc.time()[3]
  loglikelihood.new<-9999
  diff<-1  
  j<-0
  while(diff>tol){
    j<-j+1
    cat("Iteration=",j,"\n")
    loglikelihood.old<-loglikelihood.new
    ##use optim function to find gamma_hat that maximizes the loglikelihood 
    #conditional on alpha_hat and beta_hat from the prevous iteration
    starttime_gamma<-proc.time()[3] 
    
    environment(AFT.logLik.gamma) <- environment()
    environment(AFT.logLik.gamma.der) <- environment()
    fn<-AFT.logLik.gamma;gr <- AFT.logLik.gamma.der
    par<-gammaVec
    
    method <- "BFGS";
    lower <- -Inf; upper <- Inf;
    control <- list(maxit = 5000,fnscale=-1); hessian <- FALSE;
    
    arg_list <- get_args_for(optim, dots = list())
    update_gamma<- do.call(optim, arg_list)
    
    gamma.new<-update_gamma$par
    cat("gamma=",gamma.new,"\n")
    if (update_gamma$convergence==0){
      cat("gamma convergence succeeded","\n")
    } else {
      cat("gamma convergence failed, maximum iteration had been reached","\n")  
    }
    cat("gamma_likelihood=",update_gamma$value,"\n")
    cat("time_gamma",proc.time()[3]-starttime_gamma,"\n")
    
    ##use optim function to find beta_hat that maximizes the loglikelihood 
    #conditional on alpha_hat and gamma_hat from the previous iteration
    
    starttime_beta<-proc.time()[3]
    #when there is no TD effect, and all covariates have NL effects,
    #no beta is being estimated
    if (sum(TD)==0 & sum(NL)==nvar){
      beta.new<-betaVec
    } else{
      gammaVec<-gamma.new
      environment(AFT.logLik.beta) <- environment()
      environment(AFT.logLik.beta.der) <- environment()
      fn<-AFT.logLik.beta;gr <- AFT.logLik.beta.der
      par<-betaVec
      
      arg_list <- get_args_for(optim, dots = list())
      update_beta<- do.call(optim, arg_list)
      
      beta.new<-update_beta$par
      if (update_beta$convergence==0){
        cat("beta convergence succeeded","\n")
      } else {
        cat("beta convergence failed, maximum iteration had been reached","\n")  
      }
      cat("beta_likelihood=",update_beta$value,"\n")
    }
    cat("beta=",beta.new,"\n")
    cat("time_beta",proc.time()[3]-starttime_beta,"\n")
    
    ##use optim function to find alpha_hat that maximizes the loglikelihood 
    #conditional on beta_hat and gamma_hat from the prevous iteration
    
    starttime_alpha<-proc.time()[3]
    #when there is no NL effects, no alpha is being estimated
    if (sum(NL)==0){
      alpha.new<-alphaVec
      loglikelihood.new<-update_beta$value
    } else{
      gammaVec<-gamma.new
      betaVec<-beta.new
      environment(AFT.logLik.alpha) <- environment()
      environment(AFT.logLik.alpha.der) <- environment()
      fn<-AFT.logLik.alpha;gr <- AFT.logLik.alpha.der
      par<-alphaVec
      
      arg_list <- get_args_for(optim, dots = list())
      update_alpha<- do.call(optim, arg_list)
    
      alpha.new<-update_alpha$par
      loglikelihood.new<-update_alpha$value
      if (update_alpha$convergence==0){
        cat("alpha convergence succeeded","\n")
      } else {
        cat("alpha convergence failed, maximum iteration had been reached","\n")  
      }
      cat("alpha_likelihood=",update_alpha$value,"\n")
    }
    cat("alpha=",alpha.new,"\n")
    cat("time_alpha",proc.time()[3]-starttime_alpha,"\n")
    
    
    
    gammaVec<-gamma.new
    betaVec<-beta.new
    alphaVec<-alpha.new
    #calculate the difference in loglikelihood between two iterations
    diff<-abs(loglikelihood.new-loglikelihood.old)
    
    cat("Loglik=",loglikelihood.new,"\n")
    cat("diff=",diff,"\n")
  }
  
  #convert the estimates of alpha from vector to list
  #each element in the list corresponds to one covariate
  alpha.est<-vector("list",nvar)
  for (i in 1:nvar){
    if (NL[i]==1){
      alpha.est[[i]]<-alphaVec[pos.alpha[[i]]]
      names(alpha.est[[i]])<-rep(Var[i],length(pos.alpha[[i]]))
    } else {alpha.est[[i]]<-NA
    names(alpha.est[[i]])<-Var[i]
    }
  }  
  
  #convert the estimates of beta from vector to list
  #each element in the list corresponds to one covariate  
  beta.est<-vector("list",nvar)  
  for (i in 1:nvar){
    if (TD[i]==1){
      beta.est[[i]]<-betaVec[pos.beta[[i]]]
      names(beta.est[[i]])<-rep(Var[i],length(pos.beta[[i]]))
    } else {beta.est[[i]]<-NA
    names(beta.est[[i]])<-Var[i]
    }
  } 
  
  coef.est<-rep(NA,nvar)  
  for (i in 1:nvar){
    if (TD[i]==0 & NL[i]==0){
      coef.est[i]<-betaVec[pos.beta[[i]]]
    } 
  }
  names(coef.est)<-Var
  #calculate the total df in the estimation
  df.all<-df.alpha+df.beta+df.gamma
  #calculate the running time
  elapsedtime<-unname(proc.time()[3]-starttime)
  #output
  return(list("Var"=Var,"Time.Obs"=Time.Obs,"Delta"=Delta,
              "coefficient"=coef.est,
              "spline_coef_NL"=alpha.est,"spline_coef_TD"=beta.est,"spline_coef_bh"=gammaVec,
              "NL"=NL,"TD"=TD,"nknot.NL"=nknot.NL,"nknot.TD"=nknot.TD,"nknot.bh"=nknot.bh,
              "NL.knots"=alpha.knots,"TD.knots"=beta.knots,"bh.knots"=bh.knots,
              "degree.TD"=degree.TD,"degree.NL"=degree.NL,"degree.bh"=degree.bh,
              "num.events"=num.event,"df"=df.all,"logLikelihood"=loglikelihood.new,
              "ndivision"=ndivision,
              "runtime"=elapsedtime))
}


############################# Loglikelihood Function ##############################################
#loglikelihood function for estimating alpha, required by the optim function
#alpha, has to be a vector and has to be the first input in the function
AFT.logLik.alpha<-function(alphaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  ###convert alpha and beta from a vector to a list, 
  #each element of the list corresponds to one covariate
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  ###Bst are the spline basis functions for time; bs(t)
  ###Ax are the spline basis functions for X; bs(X)
  ###TD=1 and NL=1, Bs(t)*g(x)=Bst*Ax
  ###TD=1 and NL=0, Bs(t)=Bst<-Bst*X; g(x)=Ax<-1
  ###TD=0 and NL=1, Bs(t)=Bst<-1; g(x)=Ax
  ###TD=0 and NL=0; Bs(t)=Bst<-X; g(x)=Ax<-1
  
  #assign value 1 to the corresponding components (when NL and/or TD=0),
  #obtain cumulative sum: sum{i in 1:nvar} (bi(t)g(Xi)) for multivariables
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  ### Compute the spline basis for the full dataset, which is need for log likelihood calculation
  W<-exp(Btgx)*Tt
  #ensure the boundary cover all possible W
  W.range<-c(0,W)
  BsW<-bs(W,degree=degree.bh,intercept=TRUE,knots=bh.knots,Boundary.knots=range(W.range))
  logL1<-delta%*%(Btgx+BsW%*%gamma)
  ##use numeric integration to compute the cumulative hazard
  environment(integrat) <- environment()
  logL2<-integrat("none")
  LogLik<-logL1-sum(unlist(logL2))
  
  return(list(LogLik=LogLik))
}

#first derivative of loglikelihood rwt alpha, optional for the optim function
#alpha, has to be a vector and has to be the first input in the function

AFT.logLik.alpha.der<-function(alphaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  environment(integrat) <- environment()
  Interg<-integrat("alpha")
  
  loglik.der<-rep(NA,length=df.alpha)
  for (i in 1:nvar){
    if (NL[i]==1){
      loglik.der[pos.alpha[[i]]]<-drop(delta*drop(Bt[[i]])-Interg[[i]])%*%Ax[[i]]}
  }
  
  return(LogLik.der=loglik.der)
}

#loglikelihood function for estimating beta, required by the optim function
#beta, has to be a vector and has to be the first input in the function

AFT.logLik.beta<-function(betaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  ### Compute the spline basis for the full dataset, which is need for log likelihood calculation
  W<-exp(Btgx)*Tt
  W.range<-c(0,W)
  BsW<-bs(W,degree=degree.bh,intercept=TRUE,knots=bh.knots,Boundary.knots=range(W.range))
  logL1<-delta%*%(Btgx+BsW%*%gamma)
  
  environment(integrat) <- environment()
  
  logL2<-integrat("none")
  LogLik<-logL1-sum(unlist(logL2))
  
  
  return(list(LogLik=LogLik))
}

#first derivative of loglikelihood rwt beta, optional for the optim function
#beta, has to be a vector and has to be the first input in the function

AFT.logLik.beta.der<-function(betaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  environment(integrat) <- environment()
  Interg<-integrat("beta")
  Interg.matrix<-matrix(unlist(Interg),nrow=length(Tt))
  
  loglik.der<-rep(NA,length=df.beta)
  s<-0
  for (i in 1:nvar){
    if (TD[i]==1 | (TD[i]==0 & NL[i]==0)){
      for (j in 1:df.bt[i]){
        s<-s+1
        loglik.der[s]<-matrix(gx[[i]],ncol=length(Tt),nrow=1)%*%(delta*Bst[[i]][,j]-Interg.matrix[,s])}
    }
  }
  
  return(LogLik.der=loglik.der)
}

#loglikelihood function for estimating gamma, required by the optim function
#gamma, has to be a vector and has to be the first input in the function

AFT.logLik.gamma<-function(gammaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  ### Compute the spline basis for the full dataset, which is need for log likelihood calculation
  W<-exp(Btgx)*Tt
  W.range<-c(0,W)
  BsW<-bs(W,degree=degree.bh,intercept=TRUE,knots=bh.knots,Boundary.knots=range(W.range))
  logL1<-delta%*%(Btgx+BsW%*%gamma)
  
  environment(integrat) <- environment()
  logL2<-integrat("none")
  
  LogLik<-logL1-sum(unlist(logL2))
  
  return(list(LogLik=LogLik))
}

#first derivative of loglikelihood rwt gamma, optional for the optim function
#gamma, has to be a vector and has to be the first input in the function

AFT.logLik.gamma.der<-function(gammaVec){
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  alpha<-vector("list",length=nvar)
  beta<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    alpha[[i]]<-alphaVec[pos.alpha[[i]]]
    beta[[i]]<-betaVec[pos.beta[[i]]]
  }
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    if (is.na(df.X[i])) {df.X[i]<-1}
    
    Bt[[i]]<-matrix(Bst[[i]],ncol=df.bt[i],byrow=FALSE)%*%matrix(beta[[i]],ncol=1)
    gx[[i]]<-matrix(Ax[[i]],ncol=df.X[i],byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  gamma<-gammaVec
  
  ### Compute the spline basis for the full dataset, which is need for log likelihood calculation
  W<-exp(Btgx)*Tt
  W.range<-c(0,W)
  BsW<-bs(W,degree=degree.bh,intercept=TRUE,knots=bh.knots,Boundary.knots=range(W.range))
  
  environment(integrat) <- environment()
  Interg<-integrat("gamma")
  Interg.matrix<-matrix(unlist(Interg),nrow=length(Tt))
  loglik.der<-drop(delta%*%BsW-apply(Interg.matrix,2,sum))
  
  return(LogLik.der=loglik.der)
}

###hazard function at u: a matrix of time, each row corresponds to an individual, and each column corresponds to the small interval dividing the entire range 
#the function to be integrated
###it is a computation function contributing to the estimation of alpha, beta, gamma, or calculate the loglikelihood function

integrand<-function(u,wrt){
  n.sam<-length(Tt)
  w<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  exp2<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  
  W.matrix<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  BSt<-vector("list",length=ncol(u))   #spline basis for beta(u)
  
  BT<-vector("list",length=nvar)  #beta(u)=BSt%beta for each covariate
  for (i in 1:nvar){
    BT[[i]]<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  }
  
  for (i in 1:ncol(u)){
    BSt[[i]]<-vector("list",nvar)  #BSt is a list has length of ncol(u); each element BSt[[i]] is also a list that contains nvar elements for the spline basis corresponding to each TD effect
  }  

  BTGX<-matrix(0,nrow=n.sam,ncol=ncol(u))  #Btgx for each time: u
  for (i in 1:ncol(u)){
    for (j in 1:nvar){
      if (TD[j]==1 & NL[j]==1){
        BSt[[i]][[j]]<-bs(u[,i],degree=degree.TD[j],intercept=TRUE,knots=beta.knots[[j]],Boundary.knots=c(0,max(Tt)))
      } else if (TD[j]==1 & NL[j]==0) {
        BSt[[i]][[j]]<-bs(u[,i],degree=degree.TD[j],intercept=TRUE,knots=beta.knots[[j]],Boundary.knots=c(0,max(Tt)))*Data[,Var[j]]
      }else if (NL[j]==1 & TD[j]==0){
        BSt[[i]][[j]]<-matrix(1)} else if (NL[j]==0 & TD[j]==0) {
          BSt[[i]][[j]]<-matrix(Data[,Var[j]],ncol=1)
        }
      BT[[j]][,i]<-drop(matrix(BSt[[i]][[j]],ncol=df.bt[j],byrow=FALSE)%*%matrix(beta[[j]],ncol=1))
      BTGX[,i]<-BTGX[,i]+drop(BT[[j]][,i])*drop(gx[[j]])
    }
  }
  ####lambda0(exp(b(u)*g(x))*u)
  W.matrix<-exp(BTGX)*u
  if (wrt=="alpha") {
    Return.ls<-vector("list",nvar)
    for (i in 1:nvar) {Return.ls[[i]]<-matrix(NA,nrow=n.sam,ncol=ncol(u))}
  }
  
  if (wrt=="beta") {
    Return.ls<-vector("list",df.beta)
    for (i in 1:df.beta) {Return.ls[[i]]<-matrix(NA,nrow=n.sam,ncol=ncol(u))}
  }
  
  if (wrt=="gamma") {
    Return.ls<-vector("list",df.gamma)
    for (i in 1:df.gamma) {Return.ls[[i]]<-matrix(NA,nrow=n.sam,ncol=ncol(u))}
  }
  
  if (wrt=="none"){
    Return.ls<-vector("list",1)
  }
  
  for (i in 1:ncol(u)){
    #spline basis functions for the baseline hazard
    Bs.bh<-bs(W.matrix[,i],degree=degree.bh,intercept=TRUE,knots=bh.knots,Boundary.knots=c(0,max(W.matrix)))
    exp2[,i]<-exp(Bs.bh%*%gamma)
    #return values that corresponds to the derivative of the log-likelihood wrt to alpha, beta, gamma, or 
    #or calculate the log-likelihood
    if (wrt=="alpha"){
      for (j in 1:nvar){  
        Return.ls[[j]][,i]<-exp(BTGX[,i])*exp2[,i]*BT[[j]][,i]}
    } else if (wrt=="beta"){
      s<-0
      for (j in 1:nvar){ 
        if (TD[[j]]==1 | (TD[[j]]==0 & NL[j]==0)){
          for (k in 1:df.bt[j]){
            s<-s+1
            Return.ls[[s]][,i]<-exp(BTGX[,i])*exp2[,i]*BSt[[i]][[j]][,k]}}  
      }
    } else if (wrt=="gamma"){
      for (k in 1:df.gamma){
        Return.ls[[k]][,i]<-exp(BTGX[,i])*exp2[,i]*Bs.bh[,k]}
    } 
  }
  if (wrt=="none"){
    Return.ls[[1]]<-exp(BTGX)*exp2
  }
  
  return(Return.ls)
}

##use numeric integration to compute the cumulative hazard
#calculation is based on matrix multiplication
integrat<-function(wrt){
  #integrate from 0 to the observed time 
  bound<-cbind(0,Tt)
  #divide the entire range to many small intervals;
  #the number of intervals is to be specified by user (ndivision argument)
  #it defines the granularity of the calculation
  #the larger the number is, the longer the estimation takes
  #we may take into account the the maximum of the event time and data granularity,i.e., how precise the event is being recorded
  #by default, the number of interval=100
  
   num_divide<-ndivision
 
  xmatrix<-t(apply(bound,1,function(x) {seq(x[1],x[2],length=num_divide)}))
  step<-apply(xmatrix,1,function(x) (x[2]-x[1]))
  
  xmatrix<-(xmatrix+step/2)[,-ncol(xmatrix)]
  #compute the value at the middle point of each interval
  environment(integrand) <- environment()
  yvalue<-integrand(xmatrix,wrt)
  #numerical integration by matrix multiplication
  value<-lapply(yvalue,function(x) apply(x*step,1,sum))
  
  return(value)
}

####################### Estimation Function: TD, NL Effects #####################################
#TD effect for binary variable
TDest_bin<-function(fit,Data,var.name,time){
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  var.index<-which(Var==var.name)
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  Bst<-bs(time,knots=fit$TD.knots[[var.index]],degree=fit$degree.TD[var.index],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
  
  est.TD<-Bst%*%matrix(fit$spline_coef_TD[[var.index]],ncol=1)
  
  return(list("time"=time,"est.TD"=est.TD))
}

#TD effect for continuous variable
TDest_con<-function(fit,Data,var.name,Q.high,Q.low,time){
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  var.index<-which(Var==var.name)
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  alpha<-as.vector(fit$spline_coef_NL[[var.index]])
  
  if (fit$NL[var.index]==1){
    X<-seq(min(Data[,var.name]),max(Data[,var.name]),length=200)
    
    X.bs<-bs(X,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
             Boundary.knots=range(Data[,var.name]))
    
    range_NL<-max(X.bs%*%matrix(alpha,ncol=1))-min(X.bs%*%matrix(alpha,ncol=1))
  } else{
    range_NL<-1
  }
  
  Bst<-bs(time,knots=fit$TD.knots[[var.index]],degree=fit$degree.TD[var.index],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
  
  est.TD<-Bst%*%matrix(fit$spline_coef_TD[[var.index]],ncol=1)*range_NL
  
  if (fit$NL[var.index]==1){
    X_high<-quantile(X,Q.high)
    X_low<-quantile(X,Q.low)
    
    X.high.bs<-bs(X_high,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
                  Boundary.knots=range(Data[,var.name]))
    
    X.high<-X.high.bs%*%matrix(alpha,ncol=1)
    
    X.low.bs<-bs(X_low,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
                 Boundary.knots=range(Data[,var.name]))
    
    X.low<-X.low.bs%*%matrix(alpha,ncol=1)
    
    
    est.TD.rescale<-ifelse(rep(X.high>X.low,length(est.TD)),est.TD,-est.TD)
  } else{
    est.TD.rescale<-est.TD
  }
  return(list("time"=time,"est.TD"=est.TD.rescale))
}


#NL effect
NLest<-function(fit,Data,var.name,ref.val,Q.high,Q.low,var.val){
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  var.index<-which(Var==var.name)
  
  alpha<-as.vector(fit$spline_coef_NL[[var.index]])
  
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  X<-var.val
  
  X.bs<-bs(X,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
           Boundary.knots=range(Data[,var.name]))
  if (fit$TD[var.index]==1){
    range_NL<-max(X.bs%*%matrix(alpha,ncol=1))-min(X.bs%*%matrix(alpha,ncol=1))
  } else{
    range_NL<-1 
  }
  ref.val.bs<-bs(ref.val,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
                 Boundary.knots=range(Data[,var.name]))
  
  X.g<-(X.bs%*%matrix(alpha,ncol=1)-rep(ref.val.bs%*%matrix(alpha,ncol=1),length(X)))/range_NL
  
  if (fit$TD[var.index]==1){
    X_high<-quantile(X,Q.high)
    X_low<-quantile(X,Q.low)
    
    X.high.bs<-bs(X_high,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
                  Boundary.knots=range(Data[,var.name]))
    
    X.high<-X.high.bs%*%matrix(alpha,ncol=1)
    
    X.low.bs<-bs(X_low,degree=fit$degree.NL[var.index],intercept=TRUE,knots=fit$NL.knots[[var.index]],
                 Boundary.knots=range(Data[,var.name]))
    
    X.low<-X.low.bs%*%matrix(alpha,ncol=1)
    
    X.g.rescale<-ifelse(rep(X.high>X.low,length(X.g)),X.g,-X.g)
  } else{
    X.g.rescale<-X.g  
  }
  return(list("X"=X,"est.NL"=X.g.rescale))
}

############################# Estimation of Hazard Function #####################################
HazardEst<-function(fit,time,cov,Data){
  NL<-fit$NL
  nknot.NL<-fit$nknot.NL
  degree.NL<-fit$degree.NL
  TD<-fit$TD
  nknot.TD<-fit$nknot.TD
  degree.TD<-fit$degree.TD
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  alpha<-fit$spline_coef_NL
  beta_TD<-fit$spline_coef_TD
  beta_coef<-fit$coefficient
  
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  nvar<-length(Var)
  n.row<-nrow(Data)
  res.gamma<-fit$spline_coef_bh
  
  Ax<-vector("list",length=nvar)
  Bst<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (NL[i]==1){
      Ax[[i]]<-bs(Data[,Var[i]],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE)
    } else {
      Ax[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*Data[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Bst[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
        Bst[[i]]<-matrix(Data[,Var[i]],ncol=1)}
  }
  
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    
    if (NL[i]==0 & TD[i]==0){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_coef[i],ncol=1)
    } else if (TD[i]==1){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)
    } else if (NL[i]==1 & TD[i]==0){
      Bt[[i]]<-matrix(1)
    }
    
    gx[[i]]<-matrix(Ax[[i]],nrow=n.row,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  
  W<-exp(Btgx)*Tt
  W.range<-c(0,W)
  
  Cov.Ax<-vector("list",length=nvar)
  Time.Bst<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (NL[i]==1){
      Cov.Ax[[i]]<-bs(cov[i],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE,Boundary.knots =range(Data[,Var[i]]))
    } else {
      Cov.Ax[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      Time.Bst[[i]]<-bs(time,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Time.Bst[[i]]<-bs(time,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*cov[i]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Time.Bst[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
        Time.Bst[[i]]<-matrix(cov[i],ncol=1)}
  }
  
  Cov.Btgx<-0
  Cov.gx<-vector("list",length=nvar)
  Cov.Bt<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (is.na(Cov.Ax[[i]][1])) {Cov.Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Time.Bst[[i]][1])) {Time.Bst[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    
    if (NL[i]==0 & TD[i]==0){
      Cov.Bt[[i]]<-matrix(as.numeric(Time.Bst[[i]]),nrow=length(time),byrow=FALSE)%*%matrix(beta_coef[i],ncol=1)
    } else if (TD[i]==1){
      Cov.Bt[[i]]<-matrix(as.numeric(Time.Bst[[i]]),nrow=length(time),byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)
    } else if (NL[i]==1 & TD[i]==0){
      Cov.Bt[[i]]<-matrix(1)
      } 
    Cov.gx[[i]]<-matrix(as.numeric(Cov.Ax[[i]]),nrow=1,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Cov.Btgx<-Cov.Btgx+drop(Cov.Bt[[i]])*drop(Cov.gx[[i]])
  }
  
  TIME<-exp(Cov.Btgx)*time
  
  BsW<-bs(TIME,degree=fit$degree.bh,intercept=TRUE,knots=fit$bh.knots,Boundary.knots=range(W.range))
  flex.hazard<-exp(c(Cov.Btgx))*exp(tcrossprod(BsW,matrix(res.gamma,nrow=1)))
  
  return(list("time"=time,"hazard"=flex.hazard))
}

############################# Estimation of Survival Function #####################################
S.integrand<-function(u)
{
  
  beta_TD<-fit$spline_coef_TD
  beta_coef<-fit$coefficient
  
  n.sam<-dim(u)[1]
  w<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  exp2<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  
  W.matrix<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  BSt<-vector("list",length=ncol(u))   #spline basis for beta(t)
  
  BT<-vector("list",length=nvar)  #beta(t)=BSt%beta for each covariate
  for (i in 1:nvar){
    BT[[i]]<-matrix(NA,nrow=n.sam,ncol=ncol(u))
  }
  
  for (i in 1:ncol(u)){
    BSt[[i]]<-vector("list",nvar)  #BSt is a list has length ncol(u); each element BSt[[i]] is also a list that contains nvar elements for the spline basis corresponding to each TD effect
  }  

  BTGX<-matrix(0,nrow=n.sam,ncol=ncol(u))  #Btgx for each time u
  for (i in 1:ncol(u)){
    for (j in 1:nvar){
      if (TD[j]==1 & NL[j]==1){
        BSt[[i]][[j]]<-bs(u[,i],degree=fit$degree.TD[j],intercept=TRUE,knots=fit$TD.knots[[j]],Boundary.knots=c(0,max(Tt)))
      } else if (TD[j]==1 & NL[j]==0) {
        BSt[[i]][[j]]<-bs(u[,i],degree=fit$degree.TD[j],intercept=TRUE,knots=fit$TD.knots[[j]],Boundary.knots=c(0,max(Tt)))*cov[,j]
      }else if (NL[j]==1 & TD[j]==0){
        BSt[[i]][[j]]<-matrix(1)} else if (NL[j]==0 & TD[j]==0) {
          BSt[[i]][[j]]<-matrix(cov[,j],ncol=1)
        }
      if (NL[j]==0 & TD[j]==0){
        BT[[j]][,i]<-drop(matrix(BSt[[i]][[j]],nrow=dim(u)[1],byrow=FALSE)%*%matrix(beta_coef[j],ncol=1))
      } else if (TD[j]==1) {
        BT[[j]][,i]<-drop(matrix(BSt[[i]][[j]],nrow=dim(u)[1],byrow=FALSE)%*%matrix(beta_TD[[j]],ncol=1))
      } else if (NL[j]==1 & TD[j]==0){
        BT[[j]][,i]<-matrix(1)
      }
      BTGX[,i]<-BTGX[,i]+drop(BT[[j]][,i])*drop(gx.ref[[j]])
    }
  }
  ####lambda0(exp(b(t)*g(x))*u)
  W.matrix<-exp(BTGX)*u
  
  Return.ls<-vector("list",1)
  
  for (i in 1:ncol(u)){
    #spline basis functions for the baseline hazard
    Bs.bh<-bs(W.matrix[,i],degree=fit$degree.bh,intercept=TRUE,knots=fit$bh.knots,Boundary.knots=range(W.range))
    exp2[,i]<-exp(Bs.bh%*%fit$spline_coef_bh)
  }
  
  Return.ls[[1]]<-exp(BTGX)*exp2
  
  return(Return.ls)
}


SurvEst<-function(fit,time,cov,Data){
  NL<-fit$NL
  nknot.NL<-fit$nknot.NL
  degree.NL<-fit$degree.NL
  TD<-fit$TD
  nknot.TD<-fit$nknot.TD
  degree.TD<-fit$degree.TD
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  alpha<-fit$spline_coef_NL
  beta_TD<-fit$spline_coef_TD
  beta_coef<-fit$coefficient
  
  ndivision<-fit$ndivision
  
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  nvar<-length(Var)
  
  cov<-matrix(cov,ncol=nvar)
  n.row<-nrow(Data)
  
  Ax<-vector("list",length=nvar)
  Ax.ref<-vector("list",length=nvar)
  Bst<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (NL[i]==1){
      Ax[[i]]<-bs(Data[,Var[i]],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE)
      Ax.ref[[i]]<-bs(cov[i],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE,Boundary.knots =range(Data[,Var[i]]))
    } else {
      Ax[[i]]<-NA
      Ax.ref[[i]]<-NA
    }
    
    if (TD[i]==1 & NL[i]==1){
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*Data[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Bst[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
      Bst[[i]]<-matrix(Data[,Var[i]],ncol=1)}
  }
  
  gx<-vector("list",length=nvar)
  gx.ref<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(Ax.ref[[i]][1])) {Ax.ref[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    
    if (NL[[i]]==0 & TD[[i]]==0){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_coef[[i]],ncol=1)
    } else if (TD[[i]]==1) {
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)
    } else if (NL[[i]]==1 & TD[[i]]==0){
      Bt[[i]]<-matrix(1)
    }
    gx[[i]]<-matrix(Ax[[i]],nrow=n.row,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    gx.ref[[i]]<-matrix(Ax.ref[[i]],nrow=1,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  
  W<-exp(Btgx)*Tt
  W.range<-c(0,W)
  
  bound<-cbind(0,time)
  
  num_divide<-ndivision
  
  
  xmatrix<-t(apply(bound,1,function(x) {seq(x[1],x[2],length=num_divide)}))
  step<-apply(xmatrix,1,function(x) (x[2]-x[1]))
  
  xmatrix<-(xmatrix+step/2)[,-ncol(xmatrix)]
  xmatrix<-matrix(xmatrix,nrow=length(time))
  
  environment(S.integrand) <- environment()
  
  yvalue<-S.integrand(xmatrix)
  #numerical integration by matrix multiplication
  if (length(time)==1){
    flex.S<-exp(-sum(yvalue[[1]]*step))
  } else{
    flex.S<-exp(-unlist(lapply(yvalue,function(x) apply(x*step,1,sum))))
  }
  return(list("time"=time,"survival"=flex.S))
}

############
###estimate time-depedent time ratio###
TDTRest<-function(fit,Data,time,cov0,cov1,pct){
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  flex.S0<-SurvEst(fit,time,cov=cov0,Data)
  flex.S1<-SurvEst(fit,time,cov=cov1,Data)
  est.St<-rbind(flex.S0$survival,flex.S1$survival)
  est.Event.time<-matrix(NA,nrow=length(pct),ncol=2) 
  for (i in 1:length(pct)){
    sq<-pct[i]  
    est.Event.time.ind<-apply(cbind(sq,est.St),1,function(x) {which(abs(x[-1]-x[1])<=0.001)[1]})
    est.Event.time[i,]<-apply(cbind(est.Event.time.ind,rbind(flex.S0$time,flex.S0$time)),1,function(x) x[x[1]+1])
  }
  TDtr<-est.Event.time[,1]/est.Event.time[,2]
  return(list("t.ref"=est.Event.time[,1],"t.index"=est.Event.time[,2],"pct"=pct,"TD_tr"=TDtr))
}


############################# linear predictor #####################################
##linear predictor at specific time: t
## sum{j: 1:J} beta_j(t)gx_j(x_j)
pred.lp<-function(fit,Data,time,newdata){
  NL<-fit$NL
  nknot.NL<-fit$nknot.NL
  degree.NL<-fit$degree.NL
  TD<-fit$TD
  nknot.TD<-fit$nknot.TD
  degree.TD<-fit$degree.TD
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  
  alpha<-fit$spline_coef_NL
  beta_TD<-fit$spline_coef_TD
  beta_coef<-fit$coefficient
  
  Tt<-Data[,Time.Obs]
  delta<-Data[,Delta]
  
  nvar<-length(Var)
  n.row<-nrow(newdata)
  
  Ax<-vector("list",length=nvar)
  Bst<-vector("list",length=nvar)
  
  times<-rep(time,n.row)
  for (i in 1:nvar){
    if (NL[i]==1){
      Ax[[i]]<-bs(newdata[,Var[i]],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE)
    } else {
      Ax[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      Bst[[i]]<-bs(times,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Bst[[i]]<-bs(times,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*newdata[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Bst[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
        Bst[[i]]<-matrix(newdata[,Var[i]],ncol=1)}
  }
  
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    
    if (NL[i]==0 & TD[i]==0){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_coef[[i]],ncol=1)
    } else if (TD[i]==1){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)  
    }else if (NL[i]==1 & TD[i]==0){
      Bt[[i]]<-matrix(1)
      }
    gx[[i]]<-matrix(Ax[[i]],nrow=n.row,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  
  return(list("time"=time,"lp"=Btgx))
}

AFTLoglike<-function(fit,Data,newdata){
  NL<-fit$NL
  nknot.NL<-fit$nknot.NL
  degree.NL<-fit$degree.NL
  TD<-fit$TD
  nknot.TD<-fit$nknot.TD
  degree.TD<-fit$degree.TD
  
  Time.Obs<-fit$Time.Obs
  Delta<-fit$Delta
  Var<-fit$Var
  ndivision<-fit$ndivision
  
  beta.knots<-fit$TD.knots
  degree.bh<-fit$degree.bh
  bh.knots<-fit$bh.knots
  
  alpha<-fit$spline_coef_NL
  
  beta_TD<-fit$spline_coef_TD
  
  beta_coef<-fit$coefficient
  
  gamma<-fit$spline_coef_bh
  
  nvar<-length(fit$Var)
  beta<-vector("list",nvar)  
  
  for (i in 1:nvar){
    if (TD[i]==1){
    beta[[i]]<-fit$spline_coef_TD[[i]]
    } else if((TD[i]==0 & NL[i]==0)){
    beta[[i]]<-fit$coefficient[i]  
    } else{
      beta[[i]]<-NA  
    }
    names(beta[[i]])<-Var[i]  
  }

    
  n.row<-nrow(Data)
  n.row.new<-nrow(newdata)
  #Observed time
  Tt<-as.matrix(Data[,Time.Obs])
  
  newTt<-as.matrix(newdata[,Time.Obs])
  #censoring indicator
  delta.new<-newdata[,Delta]
  
  #number of covariates included in the model
  nvar<-length(Var)
  
  #number of paramter for moding beta(t) for each covariate
  df.bt<-nknot.TD+degree.TD+1
  df.bt[NL==0 & TD==0]<-1
  
  df.bt.pos<-df.bt
  df.bt.pos[is.na(df.bt)]<-0
  df.bt.pos<-cumsum(df.bt.pos)
  
  #total df for TD effects
  df.beta<-max(df.bt.pos)
  
  df.gamma<-fit$nknot.bh+fit$degree.bh+1
  
  Ax<-vector("list",length=nvar)
  Bst<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (NL[i]==1){
      Ax[[i]]<-bs(Data[,Var[i]],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE)
    } else {
      Ax[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Bst[[i]]<-bs(Tt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*Data[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Bst[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
        Bst[[i]]<-matrix(Data[,Var[i]],ncol=1)}
  }
  
  gx<-vector("list",length=nvar)
  Bt<-vector("list",length=nvar)
  Btgx<-0
  
  for (i in 1:nvar){
    if (is.na(Ax[[i]][1])) {Ax[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst[[i]][1])) {Bst[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    if (is.na(beta[[i]][1])) {beta[[i]]<-1}
    if (is.na(df.bt[i])) {df.bt[i]<-1}
    
    if (NL[i]==0 & TD[i]==0){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_coef[[i]],ncol=1)
    } else if (TD[i]==1){
      Bt[[i]]<-matrix(Bst[[i]],nrow=n.row,byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)
    }else if (NL[i]==1 & TD[i]==0){
      Bt[[i]]<-matrix(1)
      }
    
    gx[[i]]<-matrix(Ax[[i]],nrow=n.row,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx<-Btgx+drop(Bt[[i]])*drop(gx[[i]])
  }
  W<-exp(Btgx)*Tt
  
  Ax.new<-vector("list",length=nvar)
  Bst.new<-vector("list",length=nvar)
  
  for (i in 1:nvar){
    if (NL[i]==1){
      Ax.new[[i]]<-bs(newdata[,Var[i]],knots=fit$NL.knots[[i]],degree=fit$degree.NL[i],intercept=TRUE,Boundary.knots =range(Data[,Var[i]]))
    } else {
      Ax.new[[i]]<-NA}
    
    if (TD[i]==1 & NL[i]==1){
      Bst.new[[i]]<-bs(newTt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))
    } else if (TD[i]==1 & NL[i]==0) {
      Bst.new[[i]]<-bs(newTt,knots=fit$TD.knots[[i]],degree=fit$degree.TD[i],intercept=TRUE,Boundary.knots =c(0,max(Tt)))*newdata[,Var[i]]
    }else if (NL[i]==1 & TD[i]==0) 
    {
      Bst.new[[i]]<-NA} else if (NL[i]==0 & TD[i]==0) {
        Bst.new[[i]]<-matrix(newdata[,Var[i]],ncol=1)}
  }
  
  gx.new<-vector("list",length=nvar)
  Bt.new<-vector("list",length=nvar)
  Btgx.new<-0
  
  for (i in 1:nvar){
    if (is.na(Ax.new[[i]][1])) {Ax.new[[i]]<-matrix(1)}
    if (is.na(alpha[[i]][1])) {alpha[[i]]<-1}
    if (is.na(Bst.new[[i]][1])) {Bst.new[[i]]<-matrix(1)}
    if (is.na(beta_TD[[i]][1])) {beta_TD[[i]]<-1}
    
    if(NL[i]==0 & TD[i]==0){
      Bt.new[[i]]<-matrix(Bst.new[[i]],nrow=n.row.new,byrow=FALSE)%*%matrix(beta_coef[[i]],ncol=1)
    } else if(TD[i]==1){
      Bt.new[[i]]<-matrix(Bst.new[[i]],nrow=n.row.new,byrow=FALSE)%*%matrix(beta_TD[[i]],ncol=1)
    } else if(NL[i]==1 & TD[i]==0){
      Bt.new[[i]]<-matrix(1)
    }
    gx.new[[i]]<-matrix(Ax.new[[i]],nrow=n.row.new,byrow=FALSE)%*%matrix(alpha[[i]],ncol=1)
    Btgx.new<-Btgx.new+drop(Bt.new[[i]])*drop(gx.new[[i]])
  }
  
  ### Compute the spline basis for the full dataset, which is need for log likelihood calculation
  W.new<-exp(Btgx.new)*newTt
  
  W.range<-c(0,W)
  
  BsW.new<-bs(W.new,degree=fit$degree.bh,intercept=TRUE,knots=fit$bh.knots,Boundary.knots=range(W.range))
  logL1<-delta.new%*%(Btgx.new+BsW.new%*%gamma)
  
  gx<-gx.new
  Tt<-newTt
  Data<-newdata
  
  
  environment(integrat) <- environment()
  logL2<-integrat(wrt="none")
  
  
  LogLik<-logL1-sum(unlist(logL2))
  
  return(list(LogLik=LogLik))
}
