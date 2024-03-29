library('rgl')
library('ElemStatLearn')
library('rpart')

## load binary classification example data
data("mixture.example")
dat <- mixture.example

## generate GL plot of mixture data
plot_mixture_data <- expression({
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
  
  ## draw Bayes (True) classification boundary
  # dat$probm <- with(dat, matrix(prob, length(px1), length(px2)))
  # dat$cls <- with(dat, contourLines(px1, px2, probm, levels=0.5))
  # pls <- lapply(dat$cls, function(p) lines3d(p$x, p$y, z=1))
  
  # ## plot marginal probability surface and decision plane
  # sfc <- surface3d(dat$px1, dat$px2, dat$prob, alpha=1.0,
  #             color="gray", specular="gray")
  # qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
  #             color="gray", lit=FALSE)
  # 
  # ## clear the surface, decision plane, and decision boundary
  # par3d(userMatrix=diag(4)); pop3d(id=sfc); pop3d(id=qds)
  # for(pl in pls) pop3d(id=pl)
  
  qds <- quads3d(x1r[c(1,2,2,1)], x2r[c(1,1,2,2)], 0.5, alpha=0.4,
                 color="gray", lit=FALSE)
})

eval(plot_mixture_data)

## rearrange training data and input grid into data frames
dat$df <- data.frame(cbind(dat$y,dat$x))
names(dat$df) <- c('y', 'x1', 'x2')
dat$dfnew <- data.frame(dat$xnew)
names(dat$dfnew) <- c('x1', 'x2')

## fit rpart tree model
fit.rpart <- rpart(y ~ x1 + x2, data=dat$df,
                   control=rpart.control(minsplit=2), method='class')
summary(fit.rpart)
plot(fit.rpart)
text(fit.rpart, xpd=NA, use.n=TRUE)

## the probabilities in each node (region)
probs.rpart <- predict(fit.rpart, dat$dfnew, type="prob")[,2]
dat$probm.rpart <- with(dat,
                        matrix(probs.rpart, length(px1), length(px2)))
dat$cls.rpart <- with(dat, contourLines(px1, px2,
                                        probm.rpart, levels=0.5))

## plot mixture data
eval(plot_mixture_data)

## plot classification boundary
pls <- lapply(dat$cls.rpart, function(p)
  lines3d(p$x, p$y, z=1, col='purple', lwd=2))

## plot probability surface 
sfc <- surface3d(dat$px1, dat$px2, probs.rpart, alpha=1.0,
                 color="gray", specular="gray")

########################
## bagging with rpart ##
########################

## fit bagged rpart tree model
fit_bag <- function(ddf, idx=1:nrow(dat)) {
  
  ## subset to resample indices
  ddf <- ddf[idx,]
  
  ## fit tree model
  fit.rpart <- rpart(y ~ x1 + x2, data=ddf[idx,],
                     control=rpart.control(minsplit=2),
                     method='class')
  
  ## get probabilities on grid of inputs
  predict(fit.rpart, dat$dfnew, type="prob")[,2]
  
}

## get predictions for 200 resamples
fit.bag <- boot(dat$df, fit_bag, R=200)

## average probabilities across resamples
probs.bag <- colMeans(fit.bag$t)

## rearrange probabilities for plotting
dat$probm.bag <- with(dat,
                      matrix(probs.bag, length(px1), length(px2)))
dat$cls.bag <- with(dat, contourLines(px1, px2,
                                      probm.bag, levels=0.5))

## plot mixture data
eval(plot_mixture_data)

## plot classification boundary
pls <- lapply(dat$cls.bag, function(p) 
  lines3d(p$x, p$y, z=1, col='purple', lwd=2))

## plot probability surface 
sfc <- surface3d(dat$px1, dat$px2, probs.bag, alpha=1.0,
                 color="gray", specular="gray")

