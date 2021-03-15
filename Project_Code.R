rm(list = ls())

library(tm)
library(tidytext)
library(dplyr)
library(e1071)
library(class)
library(randomForest)
library(mlr)
library(tidyverse)
library(caret)
library(kernlab)

#set the working directory where data folder is present
setwd('~/Newsgroups')

# creating datafame for storing the data read from files
dat.df = data.frame(class=character(),
                    docID=numeric(),
                    text = character(), 
                    stringsAsFactors = F)

# reading tand saving files data in dataframe
folderNames <- list.files()
for (i in folderNames){
  
  for (j in list.files(i)){
    
    file_name = paste(i,'/',j,sep = '')
    text = paste(readLines(file_name), collapse=" ")
    text = removePunctuation(text)
    text = stripWhitespace(text)
    
    row.df = data.frame(i, j, text, stringsAsFactors = F)
    names(row.df)<-c("class","docID","text")
    dat.df = rbind(dat.df,row.df)
  }
}

#__________ Part 1: Eploration of data set __________#

tidy_word = unnest_tokens(dat.df, word, text)
tidy_word_freq = count(tidy_word, word, sort = TRUE)

# top 200 most popular words
word_freq = as.data.frame(tidy_word_freq)
word_freq[1:200, c('word', 'n')]
w = word_freq[1:50, c('word', 'n')]

# top 200 most popular words filter tokens by length
word_freq['Word_len'] = nchar(word_freq$word)
filter_words = subset(word_freq, Word_len>3 & Word_len<21)
filter_words[1:200, c('word', 'n')]
wf = filter_words[1:50, c('word', 'n')]
## Plots for explorations
# Creating a bar plot[in ggpubr] for top 50 words.
ggplot(w, aes(x = word, y = n)) +
  ggtitle("Plot of top 50 words") +
  ylab("Count of word") +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = n), vjust = -0.3) + 
  theme(axis.text.x = element_text(angle = 90),
        plot.title = element_text(size=14,hjust = 0.5))

# Creating a bar plot[in ggpubr] for top 50 words.
ggplot(wf, aes(x = word, y = n)) +
  ggtitle("Plot of top 50 filtered words") +
  ylab("Count of word") +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = n), vjust = -0.3) + 
  theme(axis.text.x = element_text(angle = 90),
        plot.title = element_text(size=14,hjust = 0.5))


# creating a Corpus from tm library
dat_corpus = Corpus(VectorSource(dat.df$text))

# creating the document ter matrix(bag of words)
dtm = DocumentTermMatrix(dat_corpus)
dtm_df = as.data.frame(as.matrix(dtm))

#adding the response variable in the dataframe
dtm_df['y.response_var'] = as.factor(dat.df$class)

# divide the data into training and test set
n = nrow(dtm_df)
set.seed(700)
iTrn = sample(1:n, n*0.7, replace=FALSE)
dat.trn = dtm_df[iTrn,]
dat.tst = dtm_df[-iTrn,]

dat.trn = as.data.frame(dat.trn)
names(dat.trn) <- make.names(names(dat.trn))

dat.tst = as.data.frame(dat.tst)
names(dat.tst) <- make.names(names(dat.tst))

#__________ Function for performance metrics __________#
#This function calculate and return f-Score from recall and precision
f_score = function(recl, pres){
  return(2*recl*pres/(recl+pres))
}

overall.performance = function(performance_vartor, cm){
  return(sum(performance_vartor)/sum(cm))
}


#__________ Function for naive bayes __________#
# This function impliment the naive bayes model and it's Accuracy,
# recall, precission and F-score
# Note: This is not a generic function this is made for the given specific problem
NB <- function(dat.train, dat.test, response_var = dat.train$y.response_var) {
  # total number of unique words
  n.unique_word = ncol(dat.train)-1
  
  # total no of words in everyclass
  dfc1 = subset(dat.train, response_var == 'comp.sys.ibm.pc.hardware')
  n.total_word_c1 = sum(dfc1[, -ncol(dat.train)])
  d1 = n.unique_word + n.total_word_c1
  
  dfc2 = subset(dat.train, response_var == 'sci.electronics')
  n.total_word_c2 = sum(dfc2[, -ncol(dat.train)])
  d2 = n.unique_word + n.total_word_c2
  
  dfc3 = subset(dat.train, response_var == 'talk.politics.guns')
  n.total_word_c3 = sum(dfc3[, -ncol(dat.train)])
  d3 = n.unique_word + n.total_word_c3
  
  dfc4 = subset(dat.train, response_var == 'talk.politics.misc')
  n.total_word_c4 = sum(dfc4[, -ncol(dat.train)])
  d4 = n.unique_word + n.total_word_c4
  
  # calculating probabilities of individual words
  pred_prob = numeric(nrow(dat.test))
  for (r in 1:nrow(dat.test)){
    col_index = which(dat.test[r,-ncol(dat.train)] != 0)
    wp1 = wp2 = wp3 = wp4 = 0
    for (i in col_index){
      count_c1 = sum(dat.train[which(response_var == 'comp.sys.ibm.pc.hardware'), i])
      count_c2 = sum(dat.train[which(response_var == 'sci.electronics'), i])
      count_c3 = sum(dat.train[which(response_var == 'talk.politics.guns'), i])
      count_c4 = sum(dat.train[which(response_var == 'talk.politics.misc'), i])
      
      word_prob_c1 = log((count_c1+1)/d1)
      word_prob_c2 = log((count_c2+1)/d2)
      word_prob_c3 = log((count_c3+1)/d3)
      word_prob_c4 = log((count_c4+1)/d4)
      
      wp1 = wp1 + word_prob_c1
      wp2 = wp2 + word_prob_c2
      wp3 = wp3 + word_prob_c3
      wp4 = wp4 + word_prob_c4
      
    }
    pred_prob[r] = which.max(c(wp1,wp2,wp3,wp4))
  }
  tbl.nb = table(as.factor(dat.test$y.response_var),as.factor(pred_prob))
  cm = tbl.nb
  acc.nb = sum(diag(tbl.nb))/sum(tbl.nb)
  acc.nb
  
  recall_c1 = tbl.nb[1,1]/sum(tbl.nb[1,])
  recall_c2 = tbl.nb[2,2]/sum(tbl.nb[2,])
  recall_c3 = tbl.nb[3,3]/sum(tbl.nb[3,])
  recall_c4 = tbl.nb[4,4]/sum(tbl.nb[4,])
  NB.Recall = c(recall_c1, recall_c2, recall_c3, recall_c4)
  
  prec_c1 = tbl.nb[1,1]/sum(tbl.nb[,1])
  prec_c2 = tbl.nb[2,2]/sum(tbl.nb[,2])
  prec_c3 = tbl.nb[3,3]/sum(tbl.nb[,3])
  prec_c4 = tbl.nb[4,4]/sum(tbl.nb[,4])
  NB.Precision = c(prec_c1, prec_c2, prec_c3, prec_c4)
  
  F1 = f_score(recall_c1, prec_c1)
  F2 = f_score(recall_c2, prec_c2)
  F3 = f_score(recall_c3, prec_c3)
  F4 = f_score(recall_c4, prec_c4)
  NB.FScore = c(F1, F2, F3, F4)
  
  return(list(CM = cm, Accuracy = acc.nb, recall = NB.Recall, precision = NB.Precision, FScore = NB.FScore))
}

# applying naiveBayes model

NB.out = NB(dat.trn, dat.tst)
NB.out
overall_recall = overall.performance(NB.out$recall, NB.out$CM)
overall_recall
overall_precision = overall.performance(NB.out$precision, NB.out$CM)
overall_precision
overall_fscore = overall.performance(NB.out$FScore, NB.out$CM)
overall_fscore


# creating dataset for KNN

x.train = dat.trn[, -ncol(dat.trn)]
y.train = as.numeric(dat.trn$y.response_var)
x.test = dat.tst[,-ncol(dat.trn)]
y.test = as.numeric(dat.tst$y.response_var)

# applying KNN
knn.o = knn(x.train, x.test, y.train)
tbl.knn = table(y.test,knn.o)
acc.knn = sum(diag(tbl.knn))/sum(tbl.knn)
acc.knn

recall_c1 = tbl.knn[1,1]/sum(tbl.knn[1,])
recall_c2 = tbl.knn[2,2]/sum(tbl.knn[2,])
recall_c3 = tbl.knn[3,3]/sum(tbl.knn[3,])
recall_c4 = tbl.knn[4,4]/sum(tbl.knn[4,])
knn.Recall = c(recall_c1, recall_c2, recall_c3, recall_c4)

prec_c1 = tbl.knn[1,1]/sum(tbl.knn[,1])
prec_c2 = tbl.knn[2,2]/sum(tbl.knn[,2])
prec_c3 = tbl.knn[3,3]/sum(tbl.knn[,3])
prec_c4 = tbl.knn[4,4]/sum(tbl.knn[,4])
knn.Precision = c(prec_c1, prec_c2, prec_c3, prec_c4)

F1 = f_score(recall_c1, prec_c1)
F2 = f_score(recall_c2, prec_c2)
F3 = f_score(recall_c3, prec_c3)
F4 = f_score(recall_c4, prec_c4)
knn.FScore = c(F1, F2, F3, F4)

overall_recall = overall.performance(knn.Recall, tbl.knn)
overall_recall
overall_precision = overall.performance(knn.Precision, tbl.knn)
overall_precision
overall_fscore = overall.performance(knn.FScore,tbl.knn)
overall_fscore

#Random Forest

rf.out = randomForest(dat.trn$y.response_var~., data = dat.trn, ntree = 200)

rf.yhat = predict(rf.out, dat.tst, type="class")
tbl.rf = table(dat.tst$y.response_var, rf.yhat)
acc.rf = sum(diag(tbl.rf))/sum(tbl.rf)
acc.rf

recall_c1 = tbl.rf[1,1]/sum(tbl.rf[1,])
recall_c2 = tbl.rf[2,2]/sum(tbl.rf[2,])
recall_c3 = tbl.rf[3,3]/sum(tbl.rf[3,])
recall_c4 = tbl.rf[4,4]/sum(tbl.rf[4,])
rf.Recall = c(recall_c1, recall_c2, recall_c3, recall_c4)

prec_c1 = tbl.rf[1,1]/sum(tbl.rf[,1])
prec_c2 = tbl.rf[2,2]/sum(tbl.rf[,2])
prec_c3 = tbl.rf[3,3]/sum(tbl.rf[,3])
prec_c4 = tbl.rf[4,4]/sum(tbl.rf[,4])
rf.Precision = c(prec_c1, prec_c2, prec_c3, prec_c4)

F1 = f_score(recall_c1, prec_c1)
F2 = f_score(recall_c2, prec_c2)
F3 = f_score(recall_c3, prec_c3)
F4 = f_score(recall_c4, prec_c4)
rf.FScore = c(F1, F2, F3, F4)

overall_recall = overall.performance(rf.Recall, tbl.rf)
overall_recall
overall_precision = overall.performance(rf.Precision, tbl.rf)
overall_precision
overall_fscore = overall.performance(rf.FScore,tbl.rf)
overall_fscore

model_accuracy = c(NB.out$Accuracy*100, acc.knn*100, acc.rf*100)
plot(model_accuracy )

# Comparision of models 
model = c('NB', 'RF', 'KNN')
Accuracy = c(NB.out$Accuracy,acc.rf, acc.knn)
bAcc.df = data.frame(model, Accuracy)

# ploting accuracy of different models
ggplot(bAcc.df, aes(model, Accuracy)) +
  geom_linerange(
    aes(x = model, ymin = 0, ymax = Accuracy), 
    color = "lightgray", size = 1.5
  )+
  geom_point(aes(color = model), size = 2)+
  ggpubr::color_palette("jco")+
  theme_pubclean()

#__________ preprocessing the data for robust evaluation __________#

# conver to lowercase,  remover all numbers, remove stopwords 
# removePunctuation, do lemmitization and make document term matrix 
pp.dtm = DocumentTermMatrix(dat_corpus, 
                            control = list(removeNumbers = T,
                                           removePunctuation = T,
                                           stripWhitespace = T,
                                           tolower = T,
                                           stopwords = T,
                                           stemDocument = T))

# removing the sparse terms from the document term matrix
# taking threshold value as 95 columns with 95 values as 0 will be dropped.
spp.dtm = removeSparseTerms(pp.dtm, 0.99)

spp.dtm_df = as.data.frame(as.matrix(spp.dtm))

spp.dtm_df['y.response_var'] = as.factor(dat.df$class)

# divide the processed data into training and test set
n = nrow(spp.dtm)
set.seed(7)
iTrn = sample(1:n, n*0.7, replace=FALSE)
spp.dat.trn = spp.dtm_df[iTrn,]
spp.dat.tst = spp.dtm_df[-iTrn,]

spp.dat.trn = as.data.frame(spp.dat.trn)
names(spp.dat.trn) <- make.names(names(spp.dat.trn))

spp.dat.tst = as.data.frame(spp.dat.tst)
names(spp.dat.tst) <- make.names(names(spp.dat.tst))

#__________ Doing Robust Evaluation on processed data __________#

# applying naiveBayes model

pp.NB.out = NB(spp.dat.trn, spp.dat.tst)
pp.NB.out
overall_recall = overall.performance(pp.NB.out$recall, pp.NB.out$CM)
overall_recall
overall_precision = overall.performance(pp.NB.out$precision, pp.NB.out$CM)
overall_precision
overall_fscore = overall.performance(pp.NB.out$FScore, pp.NB.out$CM)
overall_fscore

##__________ Evaluation using MLR Package __________##

# Define a task
trainTask = makeClassifTask(data = spp.dat.trn, target = 'y.response_var')

##__________ Decision tree __________##
#make tree learner
L.tree <- makeLearner("classif.rpart", predict.type = "response")

#set cross validation
set.seed(1)
cv.tree <- makeResampleDesc("CV",iters = 3L)

#Search for hyperparameters
gs.tree <- makeParamSet(
  makeIntegerParam("minsplit",lower = 30, upper = 50),
  makeNumericParam("cp", lower = 0.01, upper = 0.2)
)

#do a grid search
set.seed(1)
gscontrol.tree <- makeTuneControlGrid()

#hypertune the parameters
set.seed(1)
stune.tree <- tuneParams(learner = L.tree, 
                         resampling = cv.tree, task = trainTask, 
                         par.set = gs.tree, control = gscontrol.tree, measures = acc)

#check best parameter
stune.tree$x

# final testing
set.seed(1)
tuneprm.tree <- setHyperPars(makeLearner('classif.rpart'), par.vals=stune.tree$x)
set.seed(1)
tune.tree <- mlr::train(tuneprm.tree, trainTask)

preds.tune.tree <- as.data.frame(predict(tune.tree, newdata=spp.dat.tst))

cm.tune.tree <- confusionMatrix(as.factor(preds.tune.tree$response), as.factor(preds.tune.tree$truth))
cm.tune.tree
tbl.tree <- table(as.factor(preds.tune.tree$response), as.factor(preds.tune.tree$truth))

precision.tree <- cm.tune.tree$byClass[,5]
recall.tree <- cm.tune.tree$byClass[,6]
Fscore.tree <- cm.tune.tree$byClass[,7]

overall_recall = overall.performance(recall.tree, tbl.tree)
overall_recall
overall_precision = overall.performance(precision.tree, tbl.tree)
overall_precision
overall_fscore = overall.performance(Fscore.tree, tbl.tree)
overall_fscore

##__________ Random forest __________##

#create a learner
L.rf <- makeLearner("classif.randomForest", predict.type = "response",
                    par.vals = list(ntree = 200, mtry = 3))

#set tunable parameters
#grid search to find hyperparameters
gs.rf <- makeParamSet(
  makeDiscreteParam("ntree",values = c(300, 400, 500)),
  makeIntegerParam("mtry", lower = 1, upper = 10)
)

#random search for 30 iterations
set.seed(1)
gscontrol.rf <- makeTuneControlRandom(maxit = 30L)

#set 3 fold cross validation
set.seed(1)
cv.rf <- makeResampleDesc("CV",iters = 3L)

#hypertuning
set.seed(1)
stune.rf <- tuneParams(learner = L.rf, resampling = cv.rf, task = trainTask, 
                       par.set = gs.rf, control = gscontrol.rf, measures = acc)

#check best parameter
stune.rf$x

# final testing
set.seed(1)
tuneprm.rf <- setHyperPars(makeLearner('classif.randomForest'), par.vals=stune.rf$x)
set.seed(1)
tune.rf <- mlr::train(tuneprm.rf, trainTask)

preds.tune.rf <- as.data.frame(predict(tune.rf, newdata=spp.dat.tst))

cm.tune.rf <- confusionMatrix(as.factor(preds.tune.rf$response), as.factor(preds.tune.rf$truth))
cm.tune.rf
tbl.rf <- table(as.factor(preds.tune.rf$response), as.factor(preds.tune.rf$truth))

performance.rf <- cm.tune.rf$byClass
precision.rf <- cm.tune.rf$byClass[,5]
recall.rf <- cm.tune.rf$byClass[,6]
Fscore.rf <- cm.tune.rf$byClass[,7]

overall_recall = overall.performance(recall.rf, tbl.rf)
overall_recall
overall_precision = overall.performance(precision.rf, tbl.rf)
overall_precision
overall_fscore = overall.performance(Fscore.rf, tbl.rf)
overall_fscore

##__________ SVM __________##


# creating svm Learner
L.ksvm <- makeLearner("classif.ksvm", predict.type = "response")

#set tunable parameters
#grid search to find hyperparameters
gs.ksvm <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)

#search function
set.seed(1)
gscontrol.ksvm <- makeTuneControlGrid()

#set 3 fold cross validation
set.seed(1)
cv.ksvm <- makeResampleDesc("CV",iters = 3L)

#hypertuning
set.seed(1)
stune.ksvm <- tuneParams(learner = L.ksvm, resampling = cv.ksvm, task = trainTask, 
                         par.set = gs.ksvm, control = gscontrol.ksvm, measures = acc)

#check best parameter
stune.ksvm$x

# final testing
set.seed(1)
tuneprm.ksvm <- setHyperPars(makeLearner('classif.ksvm'), par.vals=stune.ksvm$x)
set.seed(1)
tune.ksvm <- mlr::train(tuneprm.ksvm, trainTask)

preds.tune.ksvm <- as.data.frame(predict(tune.ksvm, newdata=spp.dat.tst))

cm.tune.ksvm <- confusionMatrix(as.factor(preds.tune.ksvm$response), as.factor(preds.tune.ksvm$truth))
cm.tune.ksvm$overall[1]

tbl.ksvm <- table(as.factor(preds.tune.ksvm$response), as.factor(preds.tune.ksvm$truth))

performance.ksvm <- cm.tune.ksvm$byClass
precision.ksvm <- cm.tune.ksvm$byClass[,5]
recall.ksvm <- cm.tune.ksvm$byClass[,6]
Fscore.ksvm <- cm.tune.ksvm$byClass[,7]

overall_recall = overall.performance(recall.ksvm, tbl.ksvm)
overall_recall
overall_precision = overall.performance(precision.ksvm, tbl.ksvm)
overall_precision
overall_fscore = overall.performance(Fscore.ksvm, tbl.ksvm)
overall_fscore


##__________ KNN __________##


# creating knn Learner
L.knn <- makeLearner("classif.knn", predict.type = "response")

#set tunable parameters
#grid search to find hyperparameters
gs.knn <- makeParamSet(
  makeIntegerParam("k", lower = 1, upper = 10)
)

#search function
set.seed(1)
gscontrol.knn <- makeTuneControlGrid()

#set 3 fold cross validation
set.seed(1)
cv.knn <- makeResampleDesc("CV",iters = 3L)

#hypertuning
set.seed(1)
stune.knn <- tuneParams(learner = L.knn, resampling = cv.knn, task = trainTask, 
                        par.set = gs.knn, control = gscontrol.knn, measures = acc)

#check best parameter
stune.knn$x

# final testing
set.seed(1)
tuneprm.knn <- setHyperPars(makeLearner('classif.knn'), par.vals=stune.knn$x)
set.seed(1)
tune.knn <- mlr::train(tuneprm.knn, trainTask)

preds.tune.knn <- as.data.frame(predict(tune.knn, newdata=spp.dat.tst))

cm.tune.knn <- confusionMatrix(as.factor(preds.tune.knn$response), as.factor(preds.tune.knn$truth))

performance.knn <- cm.tune.knn$byClass

tbl.knn <- table(as.factor(preds.tune.knn$response), as.factor(preds.tune.knn$truth))

performance.knn <- cm.tune.knn$byClass
precision.knn <- cm.tune.knn$byClass[,5]
recall.knn <- cm.tune.knn$byClass[,6]
Fscore.knn <- cm.tune.knn$byClass[,7]

overall_recall = overall.performance(recall.knn, tbl.knn)
overall_recall
overall_precision = overall.performance(precision.knn, tbl.knn)
overall_precision
overall_fscore = overall.performance(Fscore.knn, tbl.knn)
overall_fscore


model = c('NB', 'RF', 'KNN')
Accuracy = c(NB.out$Accuracy,acc.rf, acc.knn)
bAcc.df = data.frame(model, Accuracy)

##__________ maing plots __________##

## Creating dataframe for comparing robust models
Robust.model = c('Naive Bayes', "Random Forest", "KNN", "SVM", "Decision Tree")
Robust.Accuracy = c(pp.NB.out$Accuracy*100, cm.tune.rf$overall[1]*100, cm.tune.knn$overall[1]*100, cm.tune.ksvm$overall[1]*100, cm.tune.tree$overall[1]*100)
Robust.Accuracy = as.numeric(c('89.16', cm.tune.rf$overall[1]*100, cm.tune.knn$overall[1]*100, cm.tune.ksvm$overall[1]*100, cm.tune.tree$overall[1]*100))

robustAccuracy.df = data.frame(Robust.model, Robust.Accuracy)

# creating plot for comparing robust models using ggplot library
ggplot(robustAccuracy.df, aes(Robust.model, Robust.Accuracy)) +
  geom_linerange( aes(x = Robust.model, ymin = 0, ymax = 100), 
    color = "lightgray", size = 1.5)+
  geom_point(aes(color = Robust.model), size = 2)+
  ggpubr::color_palette("jco")+
  theme_pubclean()

## Creating dataframe for comparing robust and basic models
Model = c('Robust Naive Bayes', "Robust Random Forest", 
          "Robust KNN", "Robust SVM", "Robust Decision Tree", 
          "Basic Naive Bayes", "Basic Random Forest",
          "Basic KNN")

Model.Name = c('Naive Bayes', "Random Forest", "KNN", "SVM", 
               "Decision Tree", 'Naive Bayes', 
               "Random Forest", "KNN")
Model.Accuracy = c(robustAccuracy.df$Robust.Accuracy, bAcc.df$Accuracy)

compare.df <- data.frame(Model, Model.Accuracy, Model.Name)

# creating plot for comparing robust models using ggplot library
dev.new()
ggbarplot(compare.df, x = "Model", y = "Model.Accuracy",
          fill = "Model.Name",              
          color = "white",           
          palette = "jco",           
          sort.val = "asc",          
          sort.by.groups = TRUE,     
          x.text.angle = 90,         
          ggtheme = theme_pubclean()
)+
  font("x.text", size = 10, vjust = 0.5)



