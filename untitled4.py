# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:04:27 2020

@author: amir
"""
import glob

import hazm

import math

non_persian = ["q" , "Q" , "w" , "W" , "e" , "E" , "r","R", "t" , "T" , "t"  "T" , "y" , "Y" , "u" ,"U" ,"i" ,"I" ,"O" ,"o" ,"P" ,"{" ,"}" ,"[" ,"]" ,"a" ,"A" ,"S" ,"s" ,"d" ,"D" ,"f" ,"F" ,"g" ,"G" 
               ,"h" ,"H" ,"j" ,"J" ,"k" ,"K" ,"l" ,"L" ,";" ,"", ":" , "p" ,"z" ,"Z" ,"x" ,"X" ,"c" ,"C" ,"v" ,"V" ,"b" ,"B" ,"n" ,"N" ,"m" ,"M" ,"," ,"." ,"/" ,"~", "/" , "|" , "+" , "=" , "-" , "_"
               , "*" , "(", ")", "^", "&", "%", "$", "#", "@", "!", "<", ">", "?" ,"'" ,"0","1","2","3","4","5","6","7","8","9" , '"' ]


normalizer = hazm.Normalizer()



full_data_spam = []
full_data_ham = []

train_email_name=glob.glob("C:/Users/amir/Desktop/akbari/Spam-Filtering-For-Persian/emails/spamtraining/*.txt")
train_email_name2 = glob.glob("C:/Users/amir/Desktop/akbari/Spam-Filtering-For-Persian/emails/hamtraining/*.txt")
for i in train_email_name:
    file = open(i , 'r')
    pre = ""
    text = file.read().strip()
    for word in text:
        if(word not in non_persian):
            pre = pre + word
        else :
            pre = pre + ""
    full_data_spam.append(pre)
    file.close()
    
    
    

for i in train_email_name2:
    file = open(i , 'r')
    pre = ""
    text = file.read().strip()
    for word in text:
        if(word not in non_persian):
            pre = pre + word
        else :
            pre = pre + ""    
    full_data_ham.append(pre)
        
    file.close()






dataFrameSpam = [] 

for file in full_data_spam:
    sag2 = ""
    for word in file :
        if(word == "و"):
            if(word not in hazm.stopwords_list()):
                sag2 = sag2 + "" + word
        sag2 = sag2 + "" + word


        sag3 = sag2.replace("\n" , "")

        sag3 =sag3.split(" ")

        while("" in sag3):
            sag3.remove("")
    dataFrameSpam.append(sag3)
    
    
    
    
    
dataFrameHam = [] 

for file in full_data_ham:
    sag2 = ""
    for word in file :
        if(word == "و"):
            if(word not in hazm.stopwords_list()):
                sag2 = sag2 + "" + word
        sag2 = sag2 + "" + word


        sag3 = sag2.replace("\n" , "")

        sag3 =sag3.split(" ")

        while("" in sag3):
            sag3.remove("")
    dataFrameHam.append(sag3)






def df (word):
    count = 0
    for element in dataFrameSpam :
        if(word in element):
            count +=1
    for element in dataFrameHam :
        if(word in element):
            count += 1 
    if(count == 0):
        print('ss')
        count = count + 1
    return count





def tf (word , List) :
    if List.count(word) == 0:
       
        return 1
    else :
        return math.log(List.count(word)) + 1
    


def idf (word , N):
    return math.log(N/df(word))


def tf_idf(word , List , N):
    idf_ = idf(word , N)
    return tf(word, List) * idf_




def tf_idf_score(newEmail , listOldEmail):
    score = 0
    for word in newEmail:
       
       score = score+tf_idf(word, listOldEmail, 600)
       
    return score
        

print(tf_idf_score(dataFrameHam[1] , dataFrameHam[0]))









train = dataFrameSpam + dataFrameHam

scores = []
def get_neighbors(train, test_row, num_neighbors):
   for index, email in enumerate(train):
       score = tf_idf_score(test_row, email)
       scores.append((index , score))
    
   scores.sort(key=lambda tup: tup[1])
   
   
   
   return scores[:num_neighbors]
       




print (get_neighbors(train , train[0] , 8))







