library('magrittr')
library('dplyr')
library('rpart')
library('partykit')
library('utils')
library('manipulate')

####################
## loss functions ##
####################

## Huber loss
loss_huber <- function(f, y, delta=1)
  ifelse(abs(y-f) <= delta,
         0.5*(y-f)^2, delta*(abs(y-f) - 0.5*delta))

## squared error loss
loss_square <- function(f, y)
  (y-f)^2

## absolute error loss
loss_absolute <- function(f, y)
  abs(y-f)

## tilted absolute error loss
## tau - target quantile for prediction
loss_tilted <- function(f, y, tau=0.75)
  ifelse(y-f > 0, (y-f) * tau, (y-f) * (tau - 1))

## plot loss as functions of residual (y-f) 
curve(loss_square(0, x), from=-2, to=2,
      xlab='y - f', ylab='loss')
curve(loss_absolute(0, x), from=-2, to=2, add=T, col=2)
curve(loss_tilted(0, x, 0.75), from=-2, to=2, add=T, col=3)
curve(loss_huber(0, x), from=-2, to=2, add=T, col=4)
legend('top', c('squared','absolute','tilted 0.75','Huber'),
       col=1:4, lty=1, bty='n')

## constant prediction for given loss 
## this applies decision theory predict the
## value 'f' that minimizes the sum of loss
## for loss=loss_square, this returns mean(y)
## for loss=loss_absolute, this returns quantile(y, probs=0.5)
## for loss=loss_huber, this returns some other value
const_pred <- function(y, loss=loss_huber,
                       limits=c(-1e10,1e10), ...) {
  sum_loss <- function(f) sum(loss(f, y, ...))
  optimize(sum_loss, interval=limits)$minimum
}

## const_pred examples
y1 <- rexp(1000) ## mean = 1.000, median = 0.693
mean(y1)
const_pred(y1, loss=loss_square)
median(y1)
const_pred(y1, loss=loss_absolute)
const_pred(y1, loss=loss_huber)

###############################
## weak learner for boosting ##
###############################

## fit a stump (using squared error loss: method='anova')
stump <- function(dat, frm, maxdepth=1) {
  rpart(formula=frm, data=dat, method='anova',
        minsplit=2,minbucket=1,maxdepth=maxdepth,
        cp=0,maxcompete=0,maxsurrogate=0,
        usesurrogate=0,xval=0) %>%
    ## convert to constparty to make easier to 
    ## manipulate predictions from this model
    as.constparty
}

#########################
## mcycle data example ##
#########################

## example data from the (built-in) MASS package
y <- MASS::mcycle$accel
x <- MASS::mcycle$times
dat <- data.frame(y=y,x=x)
plot(dat$x, dat$y, 
     xlab='Time', ylab='Acceleration')

## fit a stump for illustration purposes
fit <- stump(dat, y~x)

## plot mean (squared errr loss) of data in each partition 
lines(dat$x, predict(fit))

## plot prediction using Huber loss in each partition
lines(dat$x, predict(fit, 
                     FUN=function(y, w) const_pred(y, loss_huber)),
      lty=2)

## adjust data then plot mean of adjusted data in each partition
fit$fitted$`(response)` <- fit$fitted$`(response)` - 50
lines(dat$x, predict(fit), lty=3)

## add a legend
legend('topleft', 
       c('mean', 'Huber loss', 'mean - 50'),
       lty=c(1,2,3), bty='n')


########################
## boosting functions ##
########################

## initial model that makes same prediction
## regardless of features
init_pred <- function(y, loss=loss_huber) {
  obj <- list(f = const_pred(y, loss), y=y)
  class(obj) <- c('init_pred', class(obj))
  return(obj)
}

## initial model predictions
## define a 'predict' function for the initial model
predict.init_pred <- function(obj, newdata, ...) {
  if(missing(newdata))
    return(rep(obj$f, length(obj$y)))
  return(rep(obj$f, nrow(newdata)))
}

## finite difference gradients (used to compute gradient of avg loss)
fdGrad <- function (pars, fun, ...,
                    .relStep = (.Machine$double.eps)^(1/2),
                    minAbsPar = 0) {
  
  npar <- length(pars)
  incr <- ifelse(abs(pars) <= minAbsPar, .relStep,
                 (abs(pars)-minAbsPar) * .relStep)
  
  sapply(1:npar, function(i) {
    del <- rep(0,npar)
    del[i] <- incr[i]
    (do.call(fun, list(pars+del, ...)) -
        do.call(fun, list(pars-del, ...)))/incr[i]/2
  })
}

## gradient boosting algorithm
## follows HTF Alg. 10.3
## dat - data frame
## frm - model formula to pass to `fit` i.e., `fit(dat, frm)`
## M   - number of committee members
## fit - function to fit weak learner
## loss - loss function(f, y)
## rho  - learning rate; should be in (0,1]
gradient_boost <- function(dat, frm, M=10, fit=stump, loss = loss_huber,
                           rho=0.25, progress=TRUE, ...) {
  
  ## extract outcome 
  trm <- terms(frm)
  vrs <- attr(trm, 'variables')
  rsp <- attr(trm, 'response')
  if(rsp == 0)
    stop('no response variable provided')
  y <- eval(vrs, dat)[[rsp]]
  
  ## list to store committee member information
  fits <- list()
  
  ## step 1. 
  ## fit initial model (constant prediction)
  fits[[1]] <- init_pred(y, loss)
  
  ## comptue initial predictions
  f <- predict(fits[[1]])
  
  ##initialize progress bar
  if(progress)
    pb <- txtProgressBar(min=1, max=M, initial=2, style=3)
  
  ## step 2.
  ## add committee members
  for(i in 2:M) {
    
    ## step 2.a.
    ## compute gradient of sum loss w/respect to predictions
    r <- -rho*fdGrad(f, function(f0) sum(loss(f0, y)))
    
    ## step 2.b.
    ## fit a tree to gradient values to get tree structure
    s <- stump(dat %>% mutate(y=r), frm)
    
    ## step 2.c.
    ## change the '(response)' element of the 'fitted' slot
    ## such that when we make predictions from this committee
    ## member, they're based on the residual from previous 
    ## iteration; this process would look different for
    ## classification problems, or for regression loss 
    ## functions that are not based on this type of residual
    s$fitted$`(response)` <- y-f
    
    ## step 2.d.
    ## update predictions using new committee member
    f <- f + predict(s,
                     FUN=function(y,w) const_pred(y, loss))
    
    ## add committee member to list
    fits[[i]] <- s
    
    ## update progress bar
    if(progress)
      setTxtProgressBar(pb, value = i)
  }
  
  ## close progress bar
  if(progress)
    close(pb)
  return(fits)
}

########################
## Huber loss example ##
########################

## do gradient boosting to 1000 iterations with Huber loss
fits_huber <- gradient_boost(dat, y~x, M=1000, loss=loss_huber)

## plot how number of trees affects fit
manipulate({
  
  ## plot acceleration data
  plot(dat$x, dat$y, 
       xlab='Time', ylab='Acceleration')
  legend('topleft', legend=paste0('M = ', m_sl), bty='n')
  
  ## compute predictions using 'm_sl' committee members
  x_plot <- seq(0, 60, 0.1)
  f <- rowSums(sapply(fits_huber[1:m_sl], function(fit)
    predict(fit, data.frame(x=x_plot),
            FUN=function(y,w) const_pred(y, loss_huber))))
  
  ## plot predictions
  lines(x_plot, f)
  
}, m_sl = slider(1, 1000, 1, 'M', 1))

#########################
## tilted loss example ##
#########################

## do gradient boosting to 1000 iterations with Huber loss
fits_tilted <- gradient_boost(dat, M=1000, loss=loss_tilted)

## plot how number of trees affects fit
manipulate({
  
  ## plot acceleration data
  plot(dat$x, dat$y, 
       xlab='Time', ylab='Acceleration')
  legend('topleft', legend=paste0('M = ', m_sl), bty='n')
  
  ## compute predictions using 'm_sl' committee members
  x_plot <- seq(0, 60, 0.1)
  f <- rowSums(sapply(fits_tilted[1:m_sl], function(fit)
    predict(fit, data.frame(x=x_plot),
            FUN=function(y,w) const_pred(y, loss_tilted))))
  
  ## plot predictions
  lines(x_plot, f)
  
}, m_sl = slider(1, 1000, 1, 'M', 1))


###############################
## tilted huber loss example ##
###############################

library('qrnn') ## for the 'tilted.approx' function

loss_tilted_huber <- function(f, y, tau=0.75, eps=1)
  tilted.approx(y-f, tau, eps)

## plot the tilted huber loss versus tilted loss
## in both cases predictions that are too small are worse
curve(loss_tilted_huber(0, x), from=-2, to=2,
      xlab='y-f', ylab='loss', lty=2)
curve(loss_tilted(0, x, 0.75), from=-2, to=2, add=T, lty=1)
legend('top', c('tilted 0.75','tilted Huber 0.75'),
       lty=1:2, bty='n')

## do gradient boosting to 1000 iterations with tilted Huber loss
fits_tilted_huber <- 
  gradient_boost(dat, M=1000, loss=loss_tilted_huber)

## plot how number of trees affects fit
manipulate({
  
  ## plot acceleration data
  plot(dat$x, dat$y, 
       xlab='Time', ylab='Acceleration')
  legend('topleft', legend=paste0('M = ', m_sl), bty='n')
  
  ## compute predictions using 'm_sl' committee members
  x_plot <- seq(0, 60, 0.1)
  f <- rowSums(sapply(fits_tilted_huber[1:m_sl], function(fit)
    predict(fit, data.frame(x=x_plot),
            FUN=function(y,w) const_pred(y, loss_tilted_huber))))
  
  ## plot predictions
  lines(x_plot, f)
  
}, m_sl = slider(1, 1000, 1, 'M', 1))