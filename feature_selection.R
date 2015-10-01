library(rhdf5)
library(picasso)
h5createFile("selected.h5")
for (i in 1:3){
  id<-paste("sub", i, sep="")
  h5createGroup("selected.h5", id)
  for (finger in 1:5) {
    X<-h5read("ECoG_big_data.h5",paste(id, "/train_data", sep=""))
    Y<-h5read("ECoG_big_data.h5",paste(id, "/train_clabel", sep=""))
    Xt<-h5read("ECoG_big_data.h5",paste(id, "/test_data", sep=""))
    Yt<-h5read("ECoG_big_data.h5",paste(id, "/test_clabel", sep=""))
    nl<-200
    f<-function(s){
      r<-picasso(t(X), Y[finger,], method=s, nlambda=nl, lambda.min.ratio=0.01, gamma=5)
    }
    g<-function(r){
      yp<-t(r$beta)%*%Xt+t(r$intercept)
      c<-numeric(nl)
      for (i in 1:nl) c[i]<-cor(yp[i,], Yt[finger,])
      fn<-which(c==max(c, na.rm=TRUE))[1]
      return(which(r$beta[,fn]!=0))
    }
    grp<-paste(id, "/finger", finger, sep="")
    h5createGroup("selected.h5", grp)
    for (method in c("scad", "mcp", "l1")) {
      s<-g(f(method))
      grps<-paste(grp, "/", method, sep="")
      h5write(s, "selected.h5", grps)
    }
  }
}
H5close()