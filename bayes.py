__author__ = 'lanx'
# -*- coding: utf-8 -*-

import numpy as np
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problem','help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting','stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in myVocabulary!" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 문서수  = wordVec 수
    numWords = len(trainMatrix[0]) # 전체 voca의 수 (모든 문서마다 같다. 모든 문서에 대한 벡터 표현이므로...
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 1인 Category의 수 / 전체 갯수 즉 P(C1)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0


    for i in range(numTrainDocs): #모든 문서에 대해서...
        if trainCategory[i] == 1: # 1로 분류된 문서이면..
            p1Num += trainMatrix[i] # 1번째 문서의 벡터를 p1Num에 더하기
            p1Denom += sum(trainMatrix[i]) # 모든 단어 갯수를 p1Denom에 더하기 (P1Denom은 Scala임)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom) # p1에 각각 더해준 단어수를 총 단어수로 나눔 --> 즉 확률 값 (P1 |
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0-pClass1)
    if (p1 > p0):
        return 1
    else:
        return 0


listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
wordVec = list()
for post in listOPosts:
    wordVec.append(setOfWords2Vec(myVocabList, post))

#for i in range(len(wordVec)):
#    print wordVec[i]
#print listOPosts
#print listClasses
print myVocabList
print wordVec


p0v, p1v, pAb = trainNB0(wordVec, listClasses)
print p0v
print p1v
print pAb

#print sum(listClasses)
testEntry = ['love','my','dalmation']
thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
print testEntry, 'classified as: ', classifyNB(thisDoc,p0v, p1v, pAb)

testEntry = ['stupid','garbage']
thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
print testEntry, 'classified as: ', classifyNB(thisDoc,p0v, p1v, pAb)