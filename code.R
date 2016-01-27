library(caret); library(rpart); library(rattle); library(rpart.plot)
library(randomForest); library(dplyr); library(parallel); library(doParallel)
library(ipred); library(RANN)

set.seed(2077)

cluster <- makeCluster(detectCores())
registerDoParallel(cluster)
fitControl <- trainControl(allowParallel = TRUE)

rawTr0 <- read.csv(file = "data/train.csv", stringsAsFactors = FALSE,
                  na.strings = c("", "NA", "#DIV/0!"), nrows = 1000)
summary(rawTr0)

# Use cabin numbers: the Titanic had 7 decks A through G, in increasing depth 
# within the ship.
# On each deck, cabin were numbered with increasing numbers as you moved towards
# the poop.
# We therefore need to split cabin into deck and cabNum. When there are several
# cabins occupied, we only keep the first one.


cabSplitTr1 <- rawTr0

# Some ambiguious cabin numbers, tidy up manually:
cabSplitTr1$Cabin[76] <- "G73"
cabSplitTr1$Cabin[129] <- "E69"
cabSplitTr1$Cabin[700] <- "G63"
cabSplitTr1$Cabin[716] <- "G73"

for (r in 1:nrow(cabSplitTr1)) {
    cabSplitTr1$nbrCabs[r] <- max(1, length(
        unlist(strsplit(cabSplitTr1$Cabin[r], " "))))
    cabSplitTr1$deck[r] <- ifelse(!is.na(cabSplitTr1$Cabin[r]), 
                              substr(cabSplitTr1$Cabin[r], 1, 1), NA)
    cabSplitTr1$cabNum[r] <- unlist(strsplit(cabSplitTr1$Cabin[r], "[A-Z]"))[2]
}

# Extract surnames and titles from Name:
namSplitTr2 <- cabSplitTr1
namSplitTr2$title <- ""
namSplitTr2$surname <- 33

for (r in 1:nrow(namSplitTr2)) {
    namSplitTr2$surname[r] <- unlist(strsplit(namSplitTr2$Name[r], ","))[1]
    namSplitTr2$title[r] <- unlist(strsplit(unlist(strsplit(
        namSplitTr2$Name[r], ","))[2], "[.]"))[1]
}


# Filter only numeric values for ticket number:
tktSplitTr3 <- namSplitTr2

for (r in 1:nrow(tktSplitTr3)) {
    pos=regexpr("[0123456789]*$",tktSplitTr3$Ticket[r])
    tktSplitTr3$ticketNum[r] <- substring(tktSplitTr3$Ticket[r], pos)
}

# 4 passengers with missing ticket numbers: Assign dummy values:
tktSplitTr3$ticketNum[180] <- "9999996"
tktSplitTr3$ticketNum[272] <- "9999997"
tktSplitTr3$ticketNum[303] <- "9999998"
tktSplitTr3$ticketNum[598] <- "9999999"

train1 <- tktSplitTr3

# Factorise categorical variables:
train1 <- transform(train1, Survived=as.factor(Survived), 
                    Pclass=as.factor(Pclass), Sex=as.factor(Sex), 
                    Embarked=as.factor(Embarked), deck=as.factor(deck), 
                    title=as.factor(title), cabNum=as.numeric(cabNum),
                    ticketNum=as.numeric(ticketNum), surname=as.factor(surname))
train1 <- select(train1, -PassengerId, -Name, -Ticket, -Cabin)

# We have many missing values we need to fill in. 
# Embarked:
train1$Embarked[is.na(train1$Embarked)] <- "S"

#Age: We are going to try several algorithms
train2bag <- train1

ageBagging <- bagging(Age ~ Pclass + Sex + SibSp + Parch + Fare + 
                          nbrCabs + title + surname + ticketNum,
                      data = train2bag[!is.na(train2bag$Age), ])
train2bag$Age[is.na(train2bag$Age)] <- predict(ageBagging, 
                                               train2[is.na(train2bag$Age), ])

train2Knn <- train1
ageKnn <- knn3(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + 
                          nbrCabs + title + surname + ticketNum,
                      data = train2Knn[!is.na(train2Knn$Age), ])
train2Knn$Age[is.na(train2Knn$Age)] <- predict.knn3(ageKnn, 
                                               train2Knn[is.na(train2Knn$Age), ], 
                                               type = "class")
                                              

train2Rf <- train1
ageRf <- randomForest(Age ~ Pclass + SibSp + Parch + Fare + title + ticketNum, 
               trControl = trainControl(allowParallel = TRUE, number = 5),
               data = train2Rf[!is.na(train2Rf$Age), ], importance = TRUE, 
               ntree=1000, mtry = 2)
train2Rf$Age[is.na(train2Rf$Age)] <- predict(ageRf, 
                                             train2Rf[is.na(train2Rf$Age), ])

# For now let's use train2Rf.
train3 <- train2Rf

# Cabin deck:
deckRf <- randomForest(deck ~ Pclass + SibSp + Parch + Fare + title 
                         + ticketNum + Age, 
                         trControl = trainControl(allowParallel = TRUE, 
                                                  number = 5),
                         data = train3[!is.na(train3$deck), ], 
                         importance = TRUE, ntree=1000, mtry = 4)
train3$deck[is.na(train3$deck)] <- predict(deckRf, 
                                             train3[is.na(train3$deck), ])
# Cabin number:
cabNumRf <- randomForest(cabNum ~ Pclass + SibSp + Parch + Fare + title 
                         + ticketNum + Age + deck, 
                         trControl = trainControl(allowParallel = TRUE, 
                                                  number = 5),
                         data = train3[!is.na(train3$cabNum), ], 
                         importance = TRUE, ntree = 1000, mtry = 6)
train3$cabNum[is.na(train3$cabNum)] <- predict(cabNumRf, 
                                          train3[is.na(train3$cabNum), ])

# Let's fit 2 Random Forest algorithms (with bootstrapping and k-fold 
# cross-validation):

rfModelBoot <- train(Survived ~ . , data = select(train3, -surname),
                    method = "rf", trControl = trainControl(
                        allowParallel = TRUE, number = 10), importance = TRUE, 
                    ntree = 1000)
rfModelKf <- train(Survived ~ . , data = select(train3, -surname), 
                   method = "rf", trControl = trainControl(allowParallel = TRUE,
                                                           method = "cv", 
                                                           number = 10),
                   importance = TRUE, ntree = 1000)

# Re-work test set:
test <- read.csv(file = "data/test.csv", stringsAsFactors = FALSE,
                 na.strings = c("", "NA", "#DIV/0!"), nrows = 1000)

for (r in 1:nrow(test)) {
    test$nbrCabs[r] <- max(1, length(
        unlist(strsplit(test$Cabin[r], " "))))
    test$deck[r] <- ifelse(!is.na(test$Cabin[r]), 
                                  substr(test$Cabin[r], 1, 1), NA)
    test$cabNum[r] <- unlist(strsplit(test$Cabin[r], "[A-Z]"))[2]
    test$surname[r] <- unlist(strsplit(test$Name[r], ","))[1]
    test$title[r] <- unlist(strsplit(unlist(strsplit(
        test$Name[r], ","))[2], "[.]"))[1]
    pos=regexpr("[0123456789]*$",test$Ticket[r])
    test$ticketNum[r] <- substring(test$Ticket[r], pos)
}

test <- transform(test, Pclass=as.factor(Pclass), Sex=as.factor(Sex), 
                    Embarked=as.factor(Embarked), deck=as.factor(deck), 
                    title=as.factor(title), cabNum=as.numeric(cabNum),
                    ticketNum=as.numeric(ticketNum), surname=as.factor(surname))
test <- select(test, -Name, -Ticket, -Cabin)

# Impute missing values:.

# Embarked:
test$Embarked[is.na(test$Embarked)] <- "S"

# Fare:
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)

# Title:
test$title[test$title == " Dona"] <- " Mrs"

# Age (with RF):
ageRfTest <- randomForest(Age ~ Pclass + SibSp + Parch + Fare + title + ticketNum, 
                      trControl = trainControl(allowParallel = TRUE, number = 5),
                      data = test[!is.na(test$Age), ], importance = TRUE, 
                      ntree=1000, mtry = 2)
test$Age[is.na(test$Age)] <- predict(ageRfTest, test[is.na(test$Age), ])

# Cabin deck:
deckRfTest <- randomForest(deck ~ Pclass + SibSp + Parch + Fare + title 
                       + ticketNum + Age, 
                       trControl = trainControl(allowParallel = TRUE, 
                                                number = 5),
                       data = test[!is.na(test$deck), ], 
                       importance = TRUE, ntree=1000, mtry = 4)
test$deck[is.na(test$deck)] <- predict(deckRfTest, 
                                           test[is.na(test$deck), ])

# Cabin number:
cabNumRfTest <- randomForest(cabNum ~ Pclass + SibSp + Parch + Fare + title 
                         + ticketNum + Age + deck, 
                         trControl = trainControl(allowParallel = TRUE, 
                                                  number = 5),
                         data = test[!is.na(test$cabNum), ], 
                         importance = TRUE, ntree = 1000, mtry = 6)
test$cabNum[is.na(test$cabNum)] <- predict(cabNumRfTest, 
                                               test[is.na(test$cabNum), ])

# Predict:
rfPredBoot <- predict(rfModelBoot, test)
rfPredKf <- predict(rfModelKf, test)



write.csv(data.frame(PassengerId = test$PassengerId, Survived = rfPredBoot), 
          file = "submissionBoot.csv", row.names = FALSE)

write.csv(data.frame(PassengerId = test$PassengerId, Survived = rfPredBoot), 
          file = "submissionKf.csv", row.names = FALSE)
