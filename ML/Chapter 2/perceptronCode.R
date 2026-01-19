require(sm)
set.seed(5)

N <- 50     # group size
xSep <- 0.9 # group separation on x axis
ySep <- 0.9 # group separation on y axis

class1.x1 <- runif(N, min = 0, max = 1)
class1.x2 <- runif(N, min = 0, max = 1)

class2.x1 <- runif(N, min = 0+xSep, max = 1+xSep)
class2.x2 <- runif(N, min = 0+ySep, max = 1+ySep)

myData.x1 <- c(class1.x1, class2.x1)
myData.x2 <- c(class1.x2, class2.x2)
target <- c(rep(-1,N), rep(1,N))

plot(myData.x1, myData.x2, type='n', xlab=expression(x[1]), ylab=expression(x[2]))
points(class1.x1, class1.x2, col=2, pch=19)
points(class2.x1, class2.x2, col=4, pch=19)

# seeds (initial weights)

w0 <- 1 
w1 <- 2 
w2 <- 3

maxIter <- 20      # max number of iterations
eta <- 0.01        # learning rate
theshold <- 0.1    # threshold to stop (maximum allowed misclassification rate)

for (i in 1:maxIter){
  print(paste('Iteration #', i))
  
  ## shuffle the order of data presentation
  index <- 1:(2*N)
  index <- sample(index)
  
  for (j in index){
    y.j <- w0 + w1*myData.x1[j] + w2*myData.x2[j]
    if (y.j >= 0){
      predicted.j <- 1
    }else{
      predicted.j <- -1}
    
    w0 <- w0 + eta*(target[j] - predicted.j)
    w1 <- w1 + eta*(target[j] - predicted.j)*myData.x1[j]
    w2 <- w2 + eta*(target[j] - predicted.j)*myData.x2[j]
  } # end FOR j  
  
  vector.y <- w0 + w1*myData.x1 + w2*myData.x2
  predicted <- rep(0,length(vector.y))
  predicted[vector.y >= 0] <- 1
  predicted[vector.y < 0] <- -1
  
  missclass.rate <- 1 - sum(predicted == target)/length(target)
  print(paste('Missclassification rate: ', missclass.rate))
  
  abline(a = -1.0*w0/w2, b = -1.0*w1/w2, col=3, lwd=3, lty=2)
  
  pause()
  
  if ( missclass.rate < theshold){  break  }
} # end FOR i

# highlight the resulting decision boundary
abline(a = -1.0*w0/w2, b = -1.0*w1/w2, col=1, lwd=3, lty=1)
