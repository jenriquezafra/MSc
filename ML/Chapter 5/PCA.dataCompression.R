#install.packages("jpeg")
library("jpeg")
myImage <- readJPEG("daffodils.jpg")

dim(myImage)
redChannel <- myImage[,,1]
greenChannel <- myImage[,,2]
blueChannel <- myImage[,,3]

red.pca <- prcomp(redChannel, center=FALSE, scale.=FALSE)
green.pca <- prcomp(greenChannel, center=FALSE, scale.=FALSE)
blue.pca <- prcomp(blueChannel, center=FALSE, scale.=FALSE)


nEigVal <- c(10,15,30,60,80,100,150,200, 250, 289)

#install.packages("abind")
library(abind)                  

# reconstruct the image with nEigVal eigenvectors
for (i in nEigVal) {
  fileName <- paste('Image_with_',i, '_components.jpg', sep="")
  comprImage <- abind(red.pca$x[,1:i] %*% t(red.pca$rotation[,1:i]),
                     green.pca$x[,1:i] %*% t(green.pca$rotation[,1:i]),
                     blue.pca$x[,1:i] %*% t(blue.pca$rotation[,1:i]),
                     along = 3)
  writeJPEG(comprImage, fileName)
}


# visualise the results
imagePlot <- function(fileName, plotTitle) {
  require('jpeg')
  img <- readJPEG(fileName)
  d <- dim(img)
  plot(0,0,xlim=c(0,d[2]),ylim=c(0,d[2]),xaxt='n',yaxt='n',xlab='',ylab='',bty='n')
  title(plotTitle, line = -0.5)
  rasterImage(img,0,0,d[2],d[2])
}

par(mfrow = c(2,5), mar = c(0,0,1,1))
for (i in nEigVal) {
  fileName <- paste('Image_with_',i, '_components.jpg', sep="") 
  plotTitle <- paste(i, ' components', sep="")
  imagePlot(fileName, plotTitle)
}



