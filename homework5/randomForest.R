library('randomForest')  ## fit random forest
library('dplyr')    ## data manipulation
library('magrittr') ## for '%<>%' operator
library('gpairs')   ## pairs plot
library('viridis')  ## viridis color palette

prostate <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'))

prostate %<>%
  mutate_at(c('svi','gleason'), ~as.factor(.))  

prostate_train <- prostate %>%
  filter(train == TRUE) %>%
  dplyr::select(-train)

prostate_test <- prostate %>%
  filter(train == FALSE) %>%
  dplyr::select(-train)

## plot lcavol vs lpsa
gpairs(prostate_train)

fit <- randomForest(lcavol ~ ., data=prostate_train, 
                    ntree=500, mtry=2, proximity=TRUE)

print(fit)          ## summary of fit object
plot(fit)           ## plot OOB MSE as function of # of trees
importance(fit)     ## variable importance 
varImpPlot(fit)     ## variable importance plot

## proximity plot
lcavol_fac <- factor(prostate_train$lcavol)
lcavol_pal <- viridis_pal()(nlevels(lcavol_fac))
MDSplot(fit, fac=lcavol_fac, palette=lcavol_pal)

## partial dependence plot for 'lpsa'
partialPlot(fit, pred.data=prostate_train, x.var='lpsa')

## test error
L2_loss <- function(y, yhat)
  (y-yhat)^2

error <- function(y, yhat, loss=L2_loss)
  mean(loss(y, yhat))

error(prostate_test$lcavol, 
      predict(fit, newdata=prostate_test))


