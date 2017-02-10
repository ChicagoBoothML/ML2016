######################################################################
###xor with nnet
library(nnet) #old neural net package, single hidden layer.
library(rgl) # graphic package used for 3-d spin
######################################################################
if(1) {cat("###simulate xor data\n")
set.seed(99)

##simulate (x1,x2) in 4 different regions
p11 = cbind(rnorm(n=25,mean=1,sd=0.5),rnorm(n=25,mean=1,sd=0.5))
p12 = cbind(rnorm(n=25,mean=-1,sd=0.5),rnorm(n=25,mean=1,sd=0.5))
p13 = cbind(rnorm(n=25,mean=-1,sd=0.5),rnorm(n=25,mean=-1,sd=0.5))
p14 = cbind(rnorm(n=25,mean=1,sd=0.5),rnorm(n=25,mean=-1,sd=0.5))

##y=g is 1 when x1 !=1 x2 and 0 when x1=x2
g = as.factor(c(rep(0,50),rep(1,50)))
x = rbind(p11,p13,p12,p14)

##big grid in (x1,x2) space
px1 = seq(min(x[,1]), max(x[,1]), length.out = 100)
px2 = seq(min(x[,2]), max(x[,2]), length.out = 100)
gd = expand.grid(x1=px1, x2=px2)

##store simulation in a data frame
dfd = data.frame(x=x,y=g)
names(dfd) =c("x1","x2","y")
}
######################################################################
if(1) {cat("###plot (x,y=g)\n")
par(mar=rep(2,4))
##red if g=1,  blue else
plot(x, col=ifelse(g==1, "coral", "cornflowerblue"), axes=FALSE)
box()
}
######################################################################
if(1) { cat("###fit logit, does not work!!!!\n")
lgfit = glm(y~.,dfd,family=binomial)
print(summary(lgfit)) # x's not significant 

phatl = predict(lgfit,newdata=gd,type="response")
print(summary(phatl)) #phat close to .5

phatlg = matrix(phatl, length(px1), length(px2)) #put phat is matrix to plot on grid
par(mar=rep(2,4))
contour(px1, px2, phatlg, levels=0.5, labels="", xlab="", ylab="",
#contour(px1, px2, phatlg, levels=(1:100)/101, labels="", xlab="", ylab="",
        main= "logit fit to xor data", axes=FALSE)
points(x, col=ifelse(g==1, "coral", "cornflowerblue"))
points(gd, pch=".", cex=1.2, col=ifelse(phatl>0.5, "coral", "cornflowerblue"))
box()

persp(px1,px2,phatlg)
}
######################################################################
if(1) {cat("###fit nnet\n")
##don't have to scale x's, alread on same scale
## size=2: number of hidden neurons
## decay=.1: L2 regularization, need to standardize x's

#uses random starting values for iterative optimization
set.seed(99) #misses
xnn = nnet(y~.,dfd,size=2,decay=.1)
phat = predict(xnn,gd)[,1]

set.seed(14) #works
xnn = nnet(y~.,dfd,size=2,decay=.1)
phat = predict(xnn,gd)[,1]
}
######################################################################
if(1) {cat("###plot fits\n")
phatg = matrix(phat, length(px1), length(px2))

#contour at phat=.5
par(mar=rep(2,4))
contour(px1, px2, phatg, levels=0.5, labels="", xlab="", ylab="",
#contour(px1, px2, phatg, levels=(1:100)/101, labels="", xlab="", ylab="",
        main= "neural network -- 1 hidden layer with 2 neurons", axes=FALSE)
points(x, col=ifelse(g==1, "coral", "cornflowerblue"))
points(gd, pch=".", cex=1.2, col=ifelse(phat>0.5, "coral", "cornflowerblue"))
box()

#3d plot
persp(px1,px2,phatg)

#spinnable 3d plot
plot3d(gd[,1],gd[,2],phat,xlab="x1",ylab="x2",zlab="phat")
}
######################################################################
if(1) {cat("###examine fit\n")
print(xnn) #structure of network
print(summary(xnn)) #actual coefficients

#write function using estimated coefficients
Flog = function(x) {return(exp(x)/(1+exp(x)))}

phatf = function(x1,x2) {
   z1 = 3.35 +   2.38*x1  -2.66*x2
   a1 = Flog(z1)

   z2 = -2.73 +  2.28*x1  -2.90*x2
   a2 = Flog(z2)

   a = Flog(2.54 -5.84*a1 +  6.30*a2)

   return(list(a=a,z1=z1,a1=a1,z2=z2,a2=a2))
}

##check first grid point
print(gd[1,])
print(phat[1])
print(phatf(gd[1,1],gd[1,2]))

##check all grid points
phatcheck = rep(0,nrow(gd))
for(i in 1:nrow(gd)) {
   phatcheck[i] = phatf(gd[i,1],gd[i,2])$a
}

plot(phat,phatcheck)
abline(0,1)
}
######################################################################
if(1) {cat("### look at fit on 4 points\n")
sdf = data.frame(x1=c(-1, -1, 1, 1), x2=c(-1, 1, -1, 1))
cat("x1 x2 z1 z2 a1 a2 phat\n")
for(i in 1:4) {
   fitx = phatf(sdf$x1[i],sdf$x2[i])
   print(unlist(fitx))

   cat("***\n\n")
}
}



