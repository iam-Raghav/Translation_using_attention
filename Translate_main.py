from __future__ import print_function
import numpy as np
import pandas as pd
import torch as t
import random
from sklearn.model_selection import train_test_split
from Translate_AiBrain import Brain
#from ConvClassifier import Brain

#Hyperparameters
###################################
hidden_size = 256
drop_out = 0.2
max_length = 16
cal_loss_every = 10
epoch = 50
###################################

#File Loading
###################################
filename = 'E:\AI_files\_Translation\data\eng-fra_clean_mod.txt'
df = pd.read_csv(filename,header=None,encoding = "utf-8", sep='\t')

###################################



inlang_word2ix = {}
outlang_word2ix = {0: "SOS",1: "EOS"}
outix2word = {0: "SOS",1: "EOS"}

#Creating word dictionary for both english and french words
###################################

for i in range(len(df.iloc[:,1])):
        for word in (df.iloc[i, 0].split()):
            if word not in inlang_word2ix:
                inlang_word2ix[word] = len(inlang_word2ix)
        for word in (df.iloc[i, 1].split()):
            if word not in outlang_word2ix:
                outlang_word2ix[word] = len(outlang_word2ix)
                outix2word[len(outix2word)] = word

print('Input language vocab size ',len(inlang_word2ix))
print('Output language vocab size',len(outlang_word2ix))
###################################

#splitting the data
###################################
input_train, input_test, target_train, target_test = train_test_split(df.iloc[:,0], df.iloc[:,1], test_size= 0.1, random_state=42)

input_train = input_train.values
target_train = target_train.values
input_test =input_test.values
target_test= target_test.values

###################################
#Defining the model

###################################
trans_model = Brain(len(inlang_word2ix), hidden_size, len(outlang_word2ix), drop_out, max_length)
print(len(input_train))

###################################

#Training and Testing the model
###################################
trans_model.learniter(input_train,target_train,inlang_word2ix,outlang_word2ix,epoch,cal_loss_every)
decoder_inputs, decoder_words, decoder_attentions = trans_model.test_samples(input_test,target_test,inlang_word2ix,outlang_word2ix,outix2word)
###################################
#Saving the Model
###################################
trans_model.save('E:\AI_files\_Translation')
###################################

#Printing Out the results
###################################
df1=pd.DataFrame()
out =[]
for m in range(len(input_test)):
    out.append([input_test[m], decoder_words[m], target_test[m]])
    # print('IN ',input_test[m])
    # print('Predicted ',decoder_words[m])
    # print('OUT ',target_test[m])
    print(out)
    # print(decoder_attentions[m])
df1=pd.DataFrame(out,columns = ['Input', 'Predicted', 'Actual_output'])
print(df1)
df1.to_csv('E:\AI_files\_Translation\data\out.csv', sep='\t', mode='w', index = False)
###################################






