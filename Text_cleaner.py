from __future__ import print_function
import numpy as np
import pandas as pd
import re
import unicodedata
#word_tokenize accepts a string as an input, not a file.

rawtext_filename = "E:\_Translation\data\eng-fra.txt" #KEY IN PATH OF SOURCE FILE
cleantext_filename = "E:\_Translation\data\eng-fra_clean.txt" #KEY IN PATH OF THE DESTINATION AND CLEAN TEXT FILE

max_length = 8

#File Loading
###################################

df = pd.read_csv(rawtext_filename,header=None,encoding = "utf-8", sep='\t')

###################################

#Converts text to ascii and remove unwanted special characters.

###################################
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
#Removing punctuations from the text
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


df1=pd.DataFrame()

for i in range(len(df.iloc[:,1])):
    if len(df.iloc[i,0].split()) < max_length:
        df.iloc[i, 0] = normalizeString(df.iloc[i, 0])

        df.iloc[i, 1] = normalizeString(df.iloc[i, 1])

        df1 = df1.append(df.loc[i], ignore_index= False)

df1.to_csv(cleantext_filename,sep='\t',header=False,index = False)
print("DONE...")


