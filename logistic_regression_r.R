library(rhdf5)
library(picasso)
i<-1
finger<-1
X<-h5read("ECoG_big_data.h5",paste(id, "/train_data", sep=""))
Y<-h5read("ECoG_big_data.h5",paste(id, "/train_blabel", sep=""))
Xt<-h5read("ECoG_big_data.h5",paste(id, "/test_data", sep=""))
Yt<-h5read("ECoG_big_data.h5",paste(id, "/test_blabel", sep=""))
nl<-200
r<-picasso(t(X), Y[finger,], family="binomial", method="l1", nlambda=nl, lambda.min.ratio=0.01, gamma=5)
sigmoid<-function(x) 1/(1+exp(-x))
predict<-function(r, Xt) t(sigmoid(t(r$beta)%*%Xt+t(r$intercept)))
y<-Yt[finger,]
#acc<-function(yp) {yp<-ifelse(yp>0.5,1,0);sum((yp==0)&(y==0))/2/sum(y==0)+sum((yp==1)&(y==1))/2/sum(y==1)}
accu<-function(x, y) min(x*exp(-1/10)+(y>0.5), 1)
accumulate<-function(yp) {
  init<-0
  c<-numeric(length(yp))
  for (i in 1:length(yp)) {
    c[i]<-accu(init, yp[i]) 
    init<-c[i]} 
  return(c)
}
pen1<-function(yp) mean(abs(y-yp))
pen2<-function(yp) mean((y-yp)^2)
cp1<-function(yp) pen1(accumulate(yp))
cp2<-function(yp) pen2(accumulate(yp))
c1 = apply(predict(r, Xt), 2, cp1)
c2 = apply(predict(r, Xt), 2, cp2)