library(rhdf5)
library(picasso)
id <- "sub1"
finger <- 1
X<-h5read("ECoG_big_data.h5",paste(id, "/train_data", sep=""))
Y<-h5read("ECoG_big_data.h5",paste(id, "/train_clabel", sep=""))
Xt<-h5read("ECoG_big_data.h5",paste(id, "/test_data", sep=""))
Yt<-h5read("ECoG_big_data.h5",paste(id, "/test_clabel", sep=""))
nl<-200
r<-picasso(t(X), Y[finger,], method="l1", nlambda=nl, lambda.min.ratio=0.01, gamma=5)
yp<-t(r$beta)%*%Xt+t(r$intercept)
yp2<-t(r$beta)%*%X+t(r$intercept)
c1<-numeric(nl)
for (i in 1:nl) c1[i]<-cor(yp[i,], Yt[finger,])
c2<-numeric(nl)
for (i in 1:nl) c2[i]<-cor(yp2[i,], Y[finger,])
df1 <- data.frame(lambda=1:200, c=c1, cc=rep("validation set",200))
df2 <- data.frame(lambda=1:200, c=c2, cc=rep("train set",200))
df <- rbind(df1,df2)
df$cc <- as.factor(df$cc)
ggplot(df, aes(x=lambda, y=c, col=cc))+geom_line()+geom_point()+
  xlab(expression(lambda))+
  ylab("correlation coefficient")+
  theme(axis.title=element_text(size=22, face="bold"))+
  geom_vline(xintercept = which(c1==max(c1)),linetype="dashed")