import numpy as np
import torch
import sys
import pandas as pd
from sklearn import preprocessing
# from keras_preprocessing.text import Tokenizer
import gc
gene_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0.25, 0.25, 0.25, 0.25],
}

def Get_DNA_Sequence(cell,TF_name,length):
    X_train = []
    y_train = []
    z_train = []
    zero_vector = [0., 0., 0., 0., 0., 0., 0., 0.]
    txt = []
    pos_file = open('./data/' + cell + "/" + TF_name + "_pos.fasta",'r')
    neg_file = open('./data/' + cell + "/" + TF_name + "_neg.fasta",'r')
    sample = []
    pos_num = 0
    neg_num = 0
    number = 0
    # print("READ DNA for %s" % (cell))
    content = '0'
    for line in pos_file:

        line = line.strip('\n\r').upper()
        # if len(line) < 5:
        #     break
        if line[0] == ">":
            i = 0
            if len(content)!=length:
                continue


        else:
            if i == 0:
               content = line
               i = i + 1
               continue
            else:
               content = content+line
               continue

        if len(content) > length:
            size = length
        else:
            size = len(content)
        row = np.random.randn(length, 4)
        for location, base in enumerate(range(0,size), start=0):
            row[location] = gene_map[content[base]]
        X_train.append(row)
        txt.append(content)
        pos_num = pos_num + 1
        y_train.append(1)
    row = np.random.randn(length, 4)
    for location, base in enumerate(range(0, size), start=0):
        row[location] = gene_map[content[base]]
    X_train.append(row)
    pos_num = pos_num + 1
    y_train.append(1)
    content = '0'
    for line in neg_file:
        size = 0
        line = line.strip('\n\r')
        # if len(line) < 5:
        #     break
        if line[0] == ">":
            i = 0
            if len(content) != length:
                continue

        else:
            if i == 0:
               content = line
               i = i + 1
               continue
            else:
               content = content+line
               continue

        if len(content) > length:
            size = length
        else:
            size = len(content)
        row = np.random.randn(length, 4)
        for location, base in enumerate(range(0,size), start=0):
            row[location] = gene_map[content[base]]
        X_train.append(row)
        neg_num = neg_num + 1
        y_train.append(0)
    row = np.random.randn(length, 4)
    for location, base in enumerate(range(0, size), start=0):
        row[location] = gene_map[content[base]]
    X_train.append(row)
    neg_num = neg_num + 1
    y_train.append(0)
    print("the number of positive train sample: %d" % pos_num)
    print("the number of negative train sample: %d" % neg_num)
    X_train = np.array(X_train,dtype="float32")
    y_train = np.array(y_train,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(X_train)))
    data = X_train[ shuffle_ix ]
    label = y_train[ shuffle_ix ]
    return data,label

gene_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],

}
# train = pd.concat([a,b]).iloc[:,3:-1].fillna(0)
def Get_TF_signal(cell, TF_name):
    TF_signal_pos = pd.read_table('./data/' + cell + "/" + TF_name + "_chip_seq_pos.fasta", sep=' ', header=None)
    TF_signal_neg = pd.read_table('./data/' + cell + "/" + TF_name + "_chip_seq_neg.fasta", sep=' ', header=None)
    TF_signal = pd.concat([TF_signal_pos,TF_signal_neg]).iloc[:,3:].fillna(0) #nan 变为0
    TF_signal = np.array(TF_signal,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(TF_signal)))
    TF_signal = TF_signal[shuffle_ix]
    return TF_signal

def Get_DNase_Score(cell, TF_name):
    DNase_pos = pd.read_table('./data/' + cell + "/" + TF_name + "_DNase_pos.fasta", sep=' ', header=None)
    DNase_neg = pd.read_table('./data/' + cell + "/" + TF_name + "_DNase_neg.fasta", sep=' ', header=None)
    DNase = pd.concat([DNase_pos,DNase_neg]).iloc[:,3:].fillna(0) #nan 变为0
    DNase = np.array(DNase,dtype="float32")
    np.random.seed(1)
    shuffle_ix = np.random.permutation(np.arange(len(DNase)))
    DNase = DNase[shuffle_ix]
    return DNase

def Get_Histone(cell, TF_name, histone_name, num, length):
    Histone = np.zeros((num, len(histone_name), length))
    i = 0
    for name in histone_name:
        histone_pos = pd.read_table('./data/' + cell + "/" + TF_name + "_" + name + "_pos.fasta", sep=' ', header=None)
        histone_neg = pd.read_table('./data/' + cell + "/" + TF_name + "_" + name + "_neg.fasta", sep=' ', header=None)
        histone = pd.concat([histone_pos,histone_neg]).iloc[:,3:].fillna(0)
        histone = np.array(histone, dtype="float32")
        np.random.seed(1)
        shuffle_ix = np.random.permutation(np.arange(len(histone)))
        histone = np.log10(1 + histone)
        histone = histone[shuffle_ix]
        Histone[:,i,:] = histone
        i += 1
    return Histone

