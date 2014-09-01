  # Test Mining with R 
# March 5, 2014
# StatLab@UVa Library
# Clay Ford


# Corpus/TDM Example ------------------------------------------------------

# toy example of turning text into numbers;
# this directory contains text files with one line of text each.
setwd("~/TextMining/docs")
# see files in working directory
dir()
# save files name to vector
files <- dir()
files

# read in text of files and store in a vector;
# create a vector to store content of files
allData <- rep(NA,length(files))
allData

# readLines() function reads lines out of text documents
readLines(files[1])

# One way to read in text:
# create a "for" loop that reads the lines of each file
for(i in 1:length(files)){
  allData[i] <- readLines(files[i])  
}
allData # vector; each element contains content of text files

# another way to read in text:
# use sapply with readLines
allData <- sapply(files, readLines)
allData

# load tm, a text mining package for R
# install.packages("tm")
library(tm)

# first create a Corpus, basically a database for text documents
doc.corpus <- Corpus(VectorSource(allData))
doc.corpus
# use the inspect() function to look at the corpus
inspect(doc.corpus)
inspect(doc.corpus[3])
str(doc.corpus) # see the structure of the corpus; looks like a database

# next create a basic Term-Document Matrix (rows=terms, columns=documents)
doc.tdm <- TermDocumentMatrix(doc.corpus)
inspect(doc.tdm) 
# notice...
# - all words reduced to lower-case
# - punctuation included
# - only words with 3 or more characters included (see wordLengths option in termFreq documentation)
# If we wanted to include words with two or more characters:
# doc.tdm <- TermDocumentMatrix(doc.corpus, control=list(wordLengths = c(2, Inf)))


# we usually want to apply transformations to corpus before creating TDM.
# use the tm_map() function.

# make all lower-case in advance; helps with stop words
doc.corpus <- tm_map(doc.corpus, tolower) 
inspect(doc.corpus)

# remove lower-case stop words
# see stop words: stopwords("english")
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("english")) 
inspect(doc.corpus)
# if you want to add stopwords:
# tm_map(doc.corpus, removeWords, c(stopwords("english"),"custom","words")) 

# remove punctuation
doc.corpus <- tm_map(doc.corpus, removePunctuation) 
inspect(doc.corpus)

# remove numbers
doc.corpus <- tm_map(doc.corpus, removeNumbers) 
inspect(doc.corpus)

doc.tdm <- TermDocumentMatrix(doc.corpus)
inspect(doc.tdm)
# notice...
# - fewer terms
# - punctuation and numbers gone
# - stopwords gone
# - no distinction between Iron in "Iron Mainden" and the iron for pressing clothes
# - weighting is simple term frequency

# Important to know how Corpora and TDMs are created. They will often be huge and not
# easily checked by eye.

# TWO MISCELLANEOUS ITEMS

# (1) Weighting a term-document matrix by term frequency - inverse document
# frequency (TF-IDF). Words with high term frequency should receive high 
# weight unless they also have high document frequency.
# http://en.wikipedia.org/wiki/Tf%E2%80%93idf
doc.tdm2 <- weightTfIdf(doc.tdm)
inspect(doc.tdm2)

# (2) can convert to actual numeric matrix for statistical analysis
tdm.mat <- as.matrix(doc.tdm)
tdm.mat

# tidy up
rm(list=ls())

# classification of tweets (JSON) -----------------------------------------

# STEP 1: read in twitter JSON files

# set directory with tweets
setwd("~/TextMining/json_files/")

# load RJSONIO package which allows us to read in JSON files using fromJSON() function
library(RJSONIO)
files <- dir() # vector of all json file names
files

# What does the fromJSON() do?
test <- fromJSON(files[1]) # reads in json file as a list object
test
test$text # extract the text
test$created_at # extract the date of the tweet

# we can combine operations: read json and extract text/date
fromJSON(files[1])$text
rm(test)

# "apply" the fromJSON() function to all JSON files and extract tweet and date;
# we'll use laply() from the plyr package; just like sapply(files, fromJSON), but faster
library(plyr)
tweets <- laply(files, function(x)fromJSON(x)$text, .progress = "text") 
tweets[1] # see first tweet

dates <- laply(files, function(x)fromJSON(x)$created_at, .progress = "text") 
dates[1] # see date of first tweet

# STEP 2: prep twitter data

# create variable measuring days after resignation; 
# seems to be a good predictor of relevance;
# 2012-06-08: Dragas and Kington ask for Sullivan's resignation

# convert dates to R date object
# format: "Sun Jun 17 20:20:03 +0000 2012")
dates <- as.Date(dates, format="%a %b %d %H:%M:%S %z %Y")
# see help(strftime) for conversion specifications
dates

# days after 2012-06-08
daysAfter <- as.numeric(dates - as.Date("2012-06-08"))
daysAfter

# tidy up
rm(dates, files)

# STEP 3: create DocumentTermMatrix

library(tm)
# create corpus
corpus <- Corpus(VectorSource(tweets)) 
corpus
inspect(corpus[1]) # see first tweet
# clean up corpus
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, removePunctuation) 
corpus <- tm_map(corpus, removeNumbers) 
inspect(corpus[1])

# create DTM; only keep terms that appear in at least 5 tweets
dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(5,Inf))))
dtm
inspect(dtm[1:5,1:5])
colnames(dtm) # see selected words

# convert document-term matrix (DTM) to matrix form;
# need this for analysis
X <- as.matrix(dtm)

# tidy up
rm(corpus, dtm)

# STEP 4: export sample to classify

# IMPORTANT TO SET SEED:
set.seed(12)
# extract sample (250 tweets) for classification
sample.tweets <- tweets[sample(length(tweets),250)] 
# export 250 sample tweets for indicator to be added

#### DO NOT DO THIS DURING WORKSHOP ####
write.csv(sample.tweets,"sample.tweets.csv", row.names=FALSE)
# open file in Excel and add indicator (1 = relevant, 0 = not relevant)
#### Done prior to workshop ####

# STEP 5: read in classified tweets and combine with DTM matrix

# will use DTM matrix and days after as the predictors in a model;
# read data with indicators (1 = relevant, 0 = not relevant)
# these data sampled with set.seed(12)
setwd("~/TextMining/")
stwt <- read.csv("sample.tweets.csv", header=TRUE)
head(stwt)
# need to sample same tweets from DTM!
# will use these 250 tweets to build model
set.seed(12)
sampX <- X[sample(length(tweets),250),] 
# these are the remaining tweets to be classified after model-building
set.seed(12)
restX <- X[-sample(length(tweets),250),] 

# add daysAfter to DTM matrices; it will be an additional predictor;
# the days after need to match up to their respective tweets, 
# so use the same set.seed() again
set.seed(12)
sampX <- cbind(sampX,daysAfter[sample(length(daysAfter),250)]) 
colnames(sampX)[dim(sampX)[2]] <- "days.after"
set.seed(12)
restX <- cbind(restX,daysAfter[-sample(length(daysAfter),250)]) 
colnames(restX)[dim(restX)[2]] <- "days.after"

# create a response vector for analysis
Y <- stwt$ind
sum(Y)/length(Y) # percent relevant

# now we're ready to build a model...

# STEP 6: create classification model using the Lasso

# classify using logistic regression (The Lasso)
library(glmnet)
# see Glmnet Vignette: http://www.stanford.edu/~hastie/glmnet/glmnet_alpha.html

# create training and testing sets of our 250 classified tweets
set.seed(1)
train <- sample(nrow(sampX), nrow(sampX)/2) # 50/50 split
test <- (-train) # rows not in the training set; hold this data out of model building
y.test <- Y[test] # Classifications of tweets not used in model building; need this to evaluate model performance

# fit logistic model via lasso
# alpha=1 is lasso; alpha=0 is ridge regression
# perform cross-validation and find the best lambda (ie, lambda with lowest misclassification error);
# type.measure="class" gives misclassification error rate
set.seed(1)
cv.out <- cv.glmnet(sampX[train,], factor(Y[train]), alpha=1, family="binomial", type.measure = "class")
plot(cv.out)
# what are the dotted vertical lines?
# lambda.min - value of log(lambda) that gives misclassification error rate.
# lambda.1se - largest value of log(lambda) such that error is within 1 standard error of the minimum

bestlam <- cv.out$lambda.min # best lambda as selected by cross validation
bestlam
log(bestlam) # what's in the plot
# the lasso model with lambda chosen by cross-validation contains only training data;
# re-run model with all data 
out <- glmnet(sampX, factor(Y), alpha=1, family="binomial")

# see selected coefficients
lasso.coef <- predict(out, type="coefficients", s=bestlam)
lasso.coef


# see how it works on training set
predict(out, newx = sampX, s = bestlam, type="class")
predict(out, newx = sampX, s = bestlam, type="class") == as.character(Y) # compare to Truth
sum(predict(out, newx = sampX, s = bestlam, type="class") == as.character(Y))/250 # proportion correct
# 80%

# create a confusion matrix
# pY = predicted classification (1 = relevant tweet, 0 = not relevant)
pY  <- as.numeric(predict(out, newx = sampX, s = bestlam, type="class"))
matrix(c(sum(pY==1 & Y==1)/length(Y), 
         sum(pY==0 & Y==1)/length(Y), # false negatives
         sum(pY==1 & Y==0)/length(Y), # false positives
         sum(pY==0 & Y==0)/length(Y)), 
       nrow=2, byrow=T,
       dimnames=list(c("pred=1","pred=0"),
                     c("actual=1", "actual=0")))
# more false positives than false negatives

# STEP 7: classify tweets

# now make predictions for tweets with no classification
predY <- predict(out, newx=restX, s=bestlam, type="class")

# combine prediction with tweets and make data frame
set.seed(12)
classifiedTweets <- data.frame(relevant=factor(predY), 
                               text=tweets[-sample(1:length(tweets),250)],
                               stringsAsFactors=F)
# first 5
classifiedTweets[1:5,]
# check a few at random
classifiedTweets[sample(750,5),]
# pull out just the relevant tweets
relevantTweets <- subset(classifiedTweets,subset= relevant=="1") 
# check a few at random
relevantTweets[sample(dim(relevantTweets)[1],5),]
# tidy up
rm(list=ls())


# sentiment analysis (NY Times API) ----------------------------------------
setwd("~/TextMining/")

# The New York Times API (application programming interfaces) allows you to programmatically 
# access New York Times data for use in your own applications.
# Usage is limited to 5000 requests per day
# request an API key: http://developer.nytimes.com/apps/register (you'll need to Register for NYTimes.com first)

library(rjson)

# Request comments on a story using The Community API
# Use a special URL formatted as follows:
# http://api.nytimes.com/svc/community/{version}/comments/url/{match-
# type}[.response-format]?{url=url-to-match}&[offset=int]$&[sort=newest]&api-key={your-API-key}

# And Then Steve Said, 'Let There Be an iPhone'
# By FRED VOGELSTEIN
# Published: October 4, 2013 


# In this example....
# match-type = exact-match
# If you specify exact-match, the first 25 comments associated with that URL will be returned if the URL is found.
# .response-format = .json
# url = http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html
# offset = 25
# sort = recommended

theCall <- "http://api.nytimes.com/svc/community/v2/comments/url/exact-match.json?url=
  http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html&
  offset=25&sort=recommended&api-key=85c35331a865573538990d5ddc0c505d:19:28348776"
test <- fromJSON(file=theCall)
test
# see just the first comment
test$results$comments[[1]]$commentBody

# see all comments
sapply(test$results$comments, function(x)x$commentBody)

# function to help download 25 NY Times comments at a time using API call
# url: link to page article with comments
# offset values: Positive integer, multiple of 25
# sort values: newest | oldest | recommended | replied | editors-selection
nytComments <- function(url,offset,nysort,apikey){
  x <- paste("http://api.nytimes.com/svc/community/v2/comments/url/exact-match.json?url=",url,
             "&offset=",offset,
             "&sort=",nysort,
             "&api-key=", apikey, sep="")
  y <- fromJSON(file=x)
  z <- sapply(y$results$comments,function(x)x$commentBody)
  z
}

# test it out
nytComments(url="http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html",
            offset=25,
            nysort="recommended",
            apikey="85c35331a865573538990d5ddc0c505d:19:28348776")

# get 100 comments
# loop through 25, 50, 75, 100
story <- "http://www.nytimes.com/2013/10/06/magazine/and-then-steve-said-let-there-be-an-iphone.html"
comments <- c()
for(i in seq(25,100,25)){ 
comments <- c(comments, nytComments(url=story,
                        offset=as.character(i),
                        nysort="recommended",
                        apikey="85c35331a865573538990d5ddc0c505d:19:28348776"))
}

# save(comments,file="comments.Rda")
# or skip the loop and load the following:
load("comments.Rda")


# sentiment analysis function

# implement a very simple algorithm to estimate
# sentiment, assigning a integer score by subtracting the number 
# of occurrences of negative words from that of positive words.
# Courtesy of:
# https://raw.github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/08a269765a6b185d5f3dd522c876043ba9628715/R/sentiment.R

# Load lists of positive and negative sentiment words for English
# (around 6800 words). Courtesy of Hu and Liu.
# see http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

poswords <- scan("~/TextMining/opinion-lexicon-English/positive-words.txt",
                 what="character", comment.char=";")
negwords <- scan("~/TextMining/opinion-lexicon-English/negative-words.txt",
                 what="character", comment.char=";")
# see a few of the words
poswords[sample(length(poswords),5)]
negwords[sample(length(negwords),5)]

# create a function to score sentiment for each comment
score.sentiment <- function(sentence){
      # clean up sentences with R's regex-driven global substitute, gsub():
      sentence <- gsub("<[^>]*>", " ",sentence) # remove html tags
      sentence <- gsub("[[:punct:]]", "", sentence) # Punctuation characters
      sentence <- gsub("[[:cntrl:]]", "", sentence) # Control characters
      sentence <- gsub("\\d+", "", sentence) # digits
      sentence <- tolower(sentence) # convert to lower case
      
      # split into words; \\s+ means split sentence where a space precedes a word
      word.list <- strsplit(sentence, "\\s+")
      # convert list into vector
      words <- unlist(word.list)
      
      # compare our words to the dictionaries of positive & negative terms
      # match() returns the position of the matched term or NA
      pos.matches <- match(words, poswords)
      neg.matches <- match(words, negwords)
      
      # we just want a TRUE/FALSE vector (0 and 1)
      pos.matches <- !is.na(pos.matches)
      neg.matches <- !is.na(neg.matches)
      
      # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
      score <- sum(pos.matches) - sum(neg.matches)
      score
}

# test the function
sample <- c("You're awesome and I love you",
            "I hate and hate and hate. So angry. Die!",
            "Impressed and amazed: you are peerless in your achievement of unparalleled mediocrity")

# apply function to each sample sentence in sample vector
lapply(sample, score.sentiment) 

# Score the comments and save
scores <- sapply(comments, score.sentiment)

# summaries
hist(scores)
table(scores)
barplot(table(scores))
summary(scores)


# most "positive" comment
comments[which(scores==20)]
# most "negative" comment
comments[which(scores==-5)]

# tidy up
rm(list=ls())


# clustering (web scraping) ----------------------------------------
# RUN THIS IN R, Not R Studio

setwd("~/TextMining/")

# Interested in summarizing reviews of FitBit; want to know more about the negative reviews
# http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=1

# STEP 1: develop strategy for scraping data
# read source code page 1 of reviews
test <- readLines("http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=1")

# work to get reviews
# inspecting page source shows us that reviews have a div class of "reviewText"
# find index numbers in test vector that contain "class="reviewText""
grep("class=\"reviewText\"",test)

# therefore this pulls only reviews
test[grep("class=\"reviewText\"",test)][1]

# work to get stars
# get number of stars for each review
# appears that rating is preceded by following code:
grep("margin-right:5px;\"><span class=\"swSprite s_star_", test)

# pull all lines of star code into vector
rating <- test[grep("margin-right:5px;\"><span class=\"swSprite s_star_", test)]
rating
# extract the number of stars
rating <- substr(rating,70,70)
rating


# STEP 2: write formal code to extract reviews
##############################################
# do not run during workshop; takes too long!
##############################################

reviews <- rep(NA, 5000) # vector to store reviews
ratings <- rep(NA, 5000) # vector to store ratings
n <- 1 # review counter
i <- 1 # page counter
# Loop until length(grep("This review is from:", test)) == 0
repeat{
  getPage <- readLines(paste("http://www.amazon.com/Fitbit-Wireless-Activity-Tracker-Charcoal/product-reviews/B0095PZHZE/ref=cm_cr_pr_top_link_2?ie=UTF8&pageNumber=",
                             i,sep=""))
  if(length(grep("This review is from:", getPage)) == 0) break else {
    for(j in 1:length(grep("This review is from:", getPage))){
      temp1 <- getPage[grep("margin-right:5px;\"><span class=\"swSprite s_star_", getPage)][j] # get the lines with ratings
      ratings[n] <- substr(temp1,70,70) # extract the rating
      reviews[n]  <- getPage[grep("class=\"reviewText\"", getPage)][j] # extract the review
      n <- n + 1
    }
    i <- i + 1
  }
}
# remove any missing records from each vector
reviews <- reviews[!is.na(reviews)]
ratings <- ratings[!is.na(ratings)]
# make into a data frame  
allReviews <- data.frame(review=reviews, rating=as.numeric(ratings), stringsAsFactors=FALSE)
# remove HTML tags from text of reviews using a regular expression
allReviews$review <- gsub("<[^>]*>", " ",allReviews$review) 
# regexp translation: "<" followed by anything but ">" to ">"

# save for later use
save(allReviews, file="amzReviews.Rda")

# load this for workshop
load("amzReviews.Rda") # collected 04-Mar-2014
head(allReviews)
str(allReviews)

# STEP 3: summarize and investigate

# avg customer review
mean(allReviews$rating)
# distribution of ratings
table(allReviews$rating)
barplot(table(allReviews$rating))

# let's look at the 1 star reviews
badReviews <- subset(allReviews, subset= rating=="1")
dim(badReviews)

# make Corpus and Document-Term matrix of 1-star reviews
library(tm)
revCorp <- Corpus(VectorSource(badReviews$review))
revCorp
revCorp <- tm_map(revCorp, tolower)
revCorp <- tm_map(revCorp, stripWhitespace)
revCorp <- tm_map(revCorp, removeWords, c(stopwords("english"),"fitbit", "zip"))
revCorp <- tm_map(revCorp, removePunctuation) 
revCorp <- tm_map(revCorp, removeNumbers) 

# wordcloud from corpus 
library("wordcloud")
wordcloud(revCorp,min.freq=15)
wordcloud(revCorp,min.freq=15, random.order=F, rot.per=0) # less artsy

# in the wordcloud:
# notice battery and batteries, replacement and replaced, 
# return and returned, work and works and worked;

# let's stem the words:
revCorp <- tm_map(revCorp, stemDocument) 
wordcloud(revCorp,min.freq=15, random.order=F, rot.per=0) 

# create document-term matrix; want words that appear in at least 10 documents
dtm <- DocumentTermMatrix(revCorp, control = list(bounds = list(global = c(10,Inf))))
dtm

# Find associations in a document-term matrix
# those terms from dtm which correlate with "term" more than corlimit.
findAssocs(dtm, "one", 0.2)
findAssocs(dtm, "batteri", 0.2)
findAssocs(dtm, "step", 0.2)
findAssocs(dtm, "work", 0.2)

# remove "one"
revCorp <- tm_map(revCorp, removeWords, "one")
dtm <- DocumentTermMatrix(revCorp, control = list(bounds = list(global = c(10,Inf))))


# cluster analysis: do they fall naturally into groups?
# k-means clustering - have to specify number of groups
azRev.sd <- scale(dtm) # standardize matrix
set.seed(9)
km.out <- kmeans(azRev.sd, 7, nstart=50) # try 7 groups
km.clusters <- km.out$cluster
table(km.clusters)

# add class membership to data frame
badReviews[,"km.clusters"] <- km.clusters
head(badReviews)

# how are they similar within clusters?
# write a function to examine each group and look at words 
# occuring most frequently

groupInfo <- function(x){
  group <- Corpus(VectorSource(badReviews[badReviews$km.clusters==x,"review"]))
  group <- tm_map(group, tolower)
  group <- tm_map(group, removeWords, c(stopwords("english"),"fitbit", "zip", "one"))
  group <- tm_map(group, removePunctuation) 
  group <- tm_map(group, removeNumbers) 
  group <- tm_map(group, stemDocument) 
  tdm <- TermDocumentMatrix(group)
  m <- as.matrix(tdm)
  v <- sort(rowSums(m),decreasing=TRUE)
  v[1:20] # top 20 most frequent terms
}

table(km.clusters)
lapply(1:7,groupInfo)


#############################################################################
#
# Bonus material: How to use the Twitter API with R

# install.packages("twitteR")
library(twitteR)

# necessary step for Windows to handle the SSL Part
# download to working directory:
download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.pem")

# Use the OAuthFactory to setup the Credentials and start accessing data in the following way
# as of 14-Jan-2014, those URLs need to be https!
reqURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"

# how to get these:
# The first step is to create a Twitter application for yourself. 
# Go to https://twitter.com/apps/new and and log in. 
# After filling in the basic info, go to the "Settings" tab and 
# select "Read, Write and Access direct messages". 
# Make sure to click on the save button after doing this. 
# In the "Details" tab, take note of your consumer key and consumer secret.

consumerKey <- "<your consumer key>"
consumerSecret <- "<your consumer secret>"
twitCred <- OAuthFactory$new(consumerKey=consumerKey,consumerSecret=consumerSecret,
                             requestURL=reqURL,accessURL=accessURL,authURL=authURL)

# Create a handshake with twitter.
# copy and paste the URL that appears into your web browser address bar and go to it;
# get the 7-digit PIN, type in and click Enter
twitCred$handshake(cainfo="cacert.pem")
registerTwitterOAuth(twitCred) # should see TRUE

# download tweets! search Twitter based on a supplied search string;
# permitted 350 requests per hour, limited to 1500 results each time;
# do not forget using your CA Cert, otherwise you will get an error message. 
tweets <- searchTwitter('@delta', cainfo="cacert.pem", n=100)



###########################################
# save the file so you don't have to do all the above again in your next session
save(twitCred, file="twitCred.Rdata")

# code to use saved file in your next R session:
load("twitCred.Rdata")
library(twitteR)
registerTwitterOAuth(twitCred)


###########################################
# Reference:
# Getting Started with TwitteR package
# http://sivaanalytics.wordpress.com/2013/10/07/getting-started-with-twitter-package/





#############################################################################
#############################################################################
#############################################################################