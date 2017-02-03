
## function to generate some data
genDataLinreg <- function(w, n) {
  thr = 0.3
  
  d = length(w)
  Result = matrix(0, n, d + 1)
  
  for (i in 1:n) {
    x = c(1, runif(d - 1, -3, 3))
    z = t(w) %*% x
    pr = exp(z) / (1+exp(z))
    if (pr > 0.5+thr) {
      pr = 1
    } else if (pr < 0.5 - thr) {
      pr = 0
    } 
    y = rbinom(1, 1, pr)
    Result[i, 1:d] = x
    Result[i, d+1] = y
  }
  
  Result
}

set.seed(10)
w = c(0.3, 0.3, -1)
n = 1000

X = genDataLinreg(w, n)
o = glm.fit(x=X[,1:3], y=X[,4], family=binomial(), intercept=FALSE)
#str(o)

# computes the loss when predicting
default_loss = function(b1, b2) {
  cw = c(o$coefficients[1], b1, b2)
  z = X[,1:3] %*% cw
  -sum((1-X[,4]) * log(1 - 1 / (1 + exp(-z) )) + X[,4] * log( 1 / (1 + exp(-z)) )) / nrow(X)
}

grad_default_loss = function(b1, b2) {
  cw = c(o$coefficients[1], b1, b2)
  z = X[,1:3] %*% cw
  phat = 1 / (1 + exp(-z))
  c( sum((phat - X[,4]) * X[,2]) / nrow(df), sum((phat - X[,4]) * X[,3]) / nrow(X) )  
}

v_default_loss = Vectorize(default_loss)

b1 = seq(0, 1, length.out = 50)
b2 = seq(-2, -1, length.out = 50)
zz = outer(b1, b2, v_default_loss)
par(mfrow = c(1, 2))
persp(b1, b2, zz, xlab = "b1", ylab="b2", zlab="J(b)")
contour(b1, b2, zz, nlevels = 15, xlab = "b1", ylab="b2")

cat ("Press [enter] to continue")
line <- readline()
graphics.off()

library(animation)
par(mar = c(4, 4, 2, 0.1))
oopt = ani.options(interval = 0.3, nmax = 10000)
out = grad.desc(FUN = v_default_loss, 
          rg=c(0, -2, 1, -1), 
          init=c(1,-1), 
          gamma=1.3,
          tol = 0.00001,
          gr=grad_default_loss,
          main="Logistic regression loss"
)
ani.options(oopt)



