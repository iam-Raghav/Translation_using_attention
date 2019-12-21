from __future__ import print_function
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#This AI model follows encoder - decoder architecture using Attention
#Encoder
#Encoder essentially encodes the input language embeddings into a GRU architecture and
#outputs the GRU output and hidden state data for each words in the input sentence.

class Trans_encoder(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(Trans_encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedded = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedded(input).view(1,1,-1) #dim 1,1,hidden_size

        output, hidden = self.gru(embedded, hidden) #dim 1,1,hidden_size
        return output, hidden

#Decoder
#Decoder here follows a seq to seq architecture where output of each word that passes through an decoder is given as the input
#to the next step of decoder.
#Attention
#Attention scores are calculated to inform the model which words of the input sentence GRU outputs needs to be attended,
#while predicting its french word counterpart.

class Trans_attnDecoder(nn.Module):
    def __init__(self,hidden_size, output_size, drop_out, max_length ):
        super(Trans_attnDecoder,self).__init__()
        self.hidden_size =hidden_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.max_length = max_length
        self.dropout = nn.Dropout(self.drop_out)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_connec = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.gru  = nn.GRU(self.hidden_size,self.hidden_size)
        self.decoder_fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outs):
        embedded = self.embedding(input).view(1,1,-1) #dim 1,1,hidden_size
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn_connec(t.cat((embedded[0],hidden[0]),1)),dim =1) #dim (1, max_length)
        attn_combined = t.bmm(attn_weights.unsqueeze(0),encoder_outs.unsqueeze(0))#dim 1, 1, hidden_size
        attn_apply = t.cat((embedded[0],attn_combined[0]),1)# dim  1, 512
        output = self.attn_fc(attn_apply).unsqueeze(0)#dim 1, 1, 256
        output = F.relu(output)
        gated_out, hidden = self.gru(output,hidden)#dim 1, 1, 256
        output = F.log_softmax(self.decoder_fc(gated_out[0]),dim=1) #dim 1, vocab_size_of_output_language
        return output, hidden, attn_weights


class Brain():
    def __init__(self,input_size, hidden_size, output_size, drop_out, max_length):
            self.max_length = max_length
            self.hidden_size = hidden_size
            self.teacher_forcing_ratio = 0.5
            self.EOS_token = 1
            self.SOS_token =0
            self.encoder = Trans_encoder(input_size,hidden_size)
            self.decoder = Trans_attnDecoder(hidden_size,output_size,drop_out,max_length)
            self.cost_fn = nn.NLLLoss()
            self.encoder_optim = optim.SGD(self.encoder.parameters(),lr=0.01)
            self.decoder_optim = optim.SGD(self.decoder.parameters(),lr=0.01)

#Preparing sequence from the dictionary.
    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return idxs

    # Running the training dataset in iteration as defined in the epochs
    def learniter(self, input_train,target_train, inlang_word2ix, outlang_word2ix,iterations, cal_loss_every):
        iter_count = 0
        iter_loss = 0
        for iter in range(iterations):
            sentence_loss = 0

            for i in range(len(input_train)):
                input_word2ix = self.prepare_sequence(input_train[i].split(), inlang_word2ix)
                input_word2ix = t.tensor(input_word2ix, dtype=t.long,)
                target_word2ix = self.prepare_sequence(target_train[i].split(), outlang_word2ix)
                target_word2ix = t.tensor(target_word2ix, dtype=t.long)
                hidden = t.zeros(1, 1, self.hidden_size, )

                loss = self.learn(input_word2ix, target_word2ix,hidden)
                sentence_loss += loss
            iter_loss += (sentence_loss/(len(input_train)))
            if iter - iter_count == cal_loss_every:
                print(sentence_loss)
                print(iter_loss/cal_loss_every)
                iter_count = iter
                iter_loss = 0

#Running the train function for single sentence
    def learn(self, input_ix_tensor, target_ix_tensor, hidden):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        len_input = input_ix_tensor.size()[0]
        len_target = target_ix_tensor.size()[0]
        encoder_outs = t.zeros(self.max_length, self.hidden_size)
        encoder_hidden = hidden
        loss =0
        for each_ix in range(len_input):
            encoder_out, encoder_hidden = self.encoder(input_ix_tensor[each_ix],encoder_hidden)
            encoder_outs[each_ix] = encoder_out[0,0]
        decoder_hidden = encoder_hidden
        decoder_input = t.tensor([[self.SOS_token]], dtype = t.long)
        #Teacher forcing is giving the actual output words in the training set as the input into the decoder instead of the words predicted by the model,
        #this helps the model to converge quickly than not using teaching forcing.
        #But this needs to be given in only for some random sentences or else the model will not learn effectively.

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for n in range(len_target):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_outs)
                target_out = target_ix_tensor[n]
                loss += self.cost_fn(decoder_output, target_out.unsqueeze(0))
                decoder_input = target_out

        else:
            for n in range(len_target):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_outs)
                decoder_input = decoder_output.max(1)[1].detach()
                target_out = target_ix_tensor[n]
                loss += self.cost_fn(decoder_output,target_out.unsqueeze(0))
                if decoder_input.item() == self.EOS_token:
                    break
        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.item()/len_target
#This function is very similar to the training iteration function, except that this is used only on the testing data with no grad enabled.
    def test_samples(self,input_train,target_train, inlang_word2ix, outlang_word2ix, outix2word):
        decorder_words=[]

        decoder_attentions=[]
        decoder_inputs = []
        for i in range(len(input_train)):
           input_word2ix = self.prepare_sequence(input_train[i].split(), inlang_word2ix)
           input_word2ix = t.tensor(input_word2ix, dtype=t.long)
           target_word2ix = self.prepare_sequence(target_train[i].split(), outlang_word2ix)
           target_word2ix = t.tensor(target_word2ix, dtype=t.long)
           hidden = t.zeros(1, 1, self.hidden_size)
           decoder_input,decorder_word, decoder_attention = self.test_sample(input_word2ix,target_word2ix, hidden,outix2word)
           decorder_words.append(decorder_word)
           decoder_attentions.append(decoder_attention)
           decoder_inputs.append(decoder_input)
        return decoder_inputs,decorder_words, decoder_attentions

#Test sample function is used similar to the train function with no teaching forcing feature and with no_grad enabled.

    def test_sample(self,input_ix_tensor, target_ix_tensor, hidden, ix2word):
        with t.no_grad():
            len_input = input_ix_tensor.size(0)
            len_target = target_ix_tensor.size(0)
            encoder_outs = t.zeros(self.max_length, self.hidden_size)
            encoder_hidden = hidden
            for each_ix in range(len_input):
                encoder_out, encoder_hidden = self.encoder(input_ix_tensor[each_ix], encoder_hidden)
                encoder_outs[each_ix] = encoder_out
            decoder_hidden = encoder_hidden
            decoder_input = t.tensor([self.SOS_token])
            decoder_attentions = t.zeros(self.max_length, self.max_length)
            decoder_words=[]
            decoder_inputs = []

            for n in range(len_target):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_outs)
                decoder_input = decoder_output.max(1)[1].detach()
                decoder_inputs.append(decoder_input)
                decoder_attentions[n] = decoder_attn
                if decoder_input.item() == self.EOS_token:
                    decoder_words.append('<EOS>')
                    break
                else:
                    decoder_words.append(ix2word[decoder_input.item()])

        return decoder_inputs,decoder_words, decoder_attentions
#this function  helps to save the model.
    def save(self,path):
        t.save(self.encoder.state_dict(), path+'\encoder.pth')
        t.save(self.decoder.state_dict(), path+'\decoder.pth')
        print('model saved')






























