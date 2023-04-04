#############################
## Basic Neural Networks in R
#############################

library('rgl')
library('ElemStatLearn')
library('nnet')
library('dplyr')

## load binary classification example data
data("mixture.example")
dat <- mixture.example

## create 3D plot of mixture data
plot_mixture_data <- function(dat=mixture.example, showtruth=FALSE) {
  ## create 3D graphic, rotate to view 2D x1/x2 projection
  par3d(FOV=1,userMatrix=diag(4))
  plot3d(dat$xnew[,1], dat$xnew[,2], dat$prob, type="n",
         xlab="x1", ylab="x2", zlab="",
         axes=FALSE, box=TRUE, aspect=1)
  ## plot points and bounding box
  x1r <- range(dat$px1)
  x2r <- range(dat$px2)
  pts <- plot3d(dat$x[,1], dat$x[,2], 1,
                type="p", radius=0.5, add=TRUE,
                col=ifelse(dat$y, "orange", "blue"))
  lns <- lines3d(x1r[c(1,2,2,1,1)], x2r[c(1,1,2,2,1)], 1)
  
  if(showtruth) {
    ## draw Bayes (True) classification boundary
    probm <- matrix(dat$prob, length(dat$px1), length(dat$px2))
    cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)
    pls <- lapply(cls, function(p) 
      lines3d(p$x, p$y, z=1, col='purple', lwd=3))
    ## plot marginal probability surface and decision plane
    sfc <- surface3d(dat$px1, dat$px2, dat$prob, alpha=1.0,
                     color="gray", specular="gray")
    qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
                   color="gray", lit=FALSE)
  }
}

## compute and plot predictions
plot_nnet_predictions <- function(fit, dat=mixture.example) {
  
  ## create figure
  plot_mixture_data()
  
  ## compute predictions from nnet
  preds <- predict(fit, dat$xnew, type="class")
  probs <- predict(fit, dat$xnew, type="raw")[,1]
  probm <- matrix(probs, length(dat$px1), length(dat$px2))
  cls <- contourLines(dat$px1, dat$px2, probm, levels=0.5)
  
  ## plot classification boundary
  pls <- lapply(cls, function(p) 
    lines3d(p$x, p$y, z=1, col='purple', lwd=2))
  
  ## plot probability surface and decision plane
  sfc <- surface3d(dat$px1, dat$px2, probs, alpha=1.0,
                   color="gray", specular="gray")
  qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
                 color="gray", lit=FALSE)
}

## plot data and 'true' probability surface
plot_mixture_data(showtruth=TRUE)

## fit single hidden layer, fully connected NN 
## 10 hidden nodes
fit <- nnet(x=dat$x, y=dat$y, size=10, entropy=TRUE, decay=0) 
plot_nnet_predictions(fit)

## count total parameters
## hidden layer w/10 nodes x (2 + 1) input nodes = 30 
## output layer w/1 node x (10 + 1) hidden nodes = 11
## 41 parameters total
length(fit$wts)

## 3 hidden nodes
nnet(x=dat$x, y=dat$y, size=3, entropy=TRUE, decay=0) %>%
  plot_nnet_predictions

## 10 hidden nodes with weight decay
nnet(x=dat$x, y=dat$y, size=10, entropy=TRUE, decay=0.02) %>%
  plot_nnet_predictions

## 3 hidden nodes with weight decay
nnet(x=dat$x, y=dat$y, size=3, entropy=TRUE, decay=0.02) %>%
  plot_nnet_predictions