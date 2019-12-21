# Neural_Translation_using_attention
Translation from english to french using encoder -decoder(seq to seq) using Attention.

# Pre-requistics
    Python- 3.7
    Torch - 1.0.0
    numpy -1.16.4
    sklearn - 0.21.3
    pandas - 0.25.0
    
# Files
1. eng-fra.txt - Contains the dataset having the English sentence and its fresh translation.
2. Text_cleaner.py - This file will clean the dataset by removing the stopwords, punctuations and converting the format to UTF-8
3. Translate_main.py - This file contains the main code that needs to be run to predict the french sentence.
4. Translate_AiBrain.py - This file contains the training, testing and Encoder decoder model required for the translation.
5. Out.csv - This file contains the input test sentence, predicted output sentence by the model and the actual output from the dataset.
This out.csv then used determine the score for any evalution metric like BLEU score.

Note:- Text cleaner file is made as a separate file for re-usability and saving time to avoid running the text cleaning process for each and every time Translate_main.py file is run.
    
# Downloads and Setup
Once you clone this repo, run the Translate_main.py file to do the sentiment analysis and to train the model.

# Evalution metric
No evalution metric is used in this translation, because there are many evalution methods available for translation and each metric has its own merits and de-merits. But for translation BLEU scoring system is widely used.
