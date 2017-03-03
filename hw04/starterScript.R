PackageList=c("MASS", 
              "ISLR",
              "animation",
              "ElemStatLearn",
              "glmnet",
              "textir",
              "nnet",
              "methods",
              "statmod",
              "stats",
              "graphics",
              "RCurl",
              "jsonlite",
              "tools",
              "utils",
              "data.table",
              "gbm",
              "ggplot2",
              "randomForest",
              "tree",
              "class",
              "kknn",
              "e1071",
              "data.table",
              "R.utils",
              "recommenderlab")
NewPackages=PackageList[!(PackageList %in% 
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)#array function

if (!("h20" %in% rownames(installed.packages()))) { 
  # Now we download, install and initialize the H2O package for R.
    install.packages("h2o", 
                     type="source", 
                     repos=(c("http://h2o-release.s3.amazonaws.com/h2o/master/3232/R")))
}


if(!file.exists("videoGames.json.gz")){
  download.file( "https://github.com/ChicagoBoothML/MachineLearning_Fall2015/raw/master/Programming%20Scripts/Lecture07/hw/videoGames.json.gz",destfile="videoGames.json.gz")
}
fileConnection <- gzcon(file("videoGames.json.gz", "rb"))
InputData = stream_in(fileConnection)


ratingData = as(InputData[c("reviewerID", "itemID", "rating")], "realRatingMatrix")
# we keep users that have rated more than 2 video games

ratingData = ratingData[rowCounts(ratingData) > 2,]

# we will focus only on popular video games that have 
# been rated by more than 3 times
ratingData = ratingData[,colCounts(ratingData) > 3]
# we are left with this many users and items
dim(ratingData)

# example on how to recommend using Popular method
r = Recommender(ratingData, method="POPULAR")

# recommend 5 items to user it row 13
rec = predict(r, ratingData[13, ], type="topNList", n=5)
as(rec, "list")

# predict ratings 
rec = predict(r, ratingData[13, ], type="ratings")
as(rec, "matrix")

