library(rhdf5)
library(picasso)
id <- "sub1"
finger <- 1
X<-h5read("ECoG_big_data.h5",paste(id, "/train_data", sep=""))
Y<-h5read("ECoG_big_data.h5",paste(id, "/train_clabel", sep=""))
nl<-200
r<-picasso(t(X), Y[finger,], method="l1", nlambda=nl, lambda.min.ratio=0.01, gamma=5)
m<-r$beta

