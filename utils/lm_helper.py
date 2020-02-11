# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:51:10 2020

@author: Shreyans
"""
import random 
import nltk
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download('mac_morpho')
from nltk.corpus import mac_morpho, brown, gutenberg



    
def get_test_data(num_brown = 200, num_mac = 20):
    test_list = []
    file_list = []
    for file in brown.fileids()[:num_brown]:
        test_list.append(list(brown.sents(file)))
        file_list.append(file)
    for file in mac_morpho.fileids()[:num_mac]:
        test_list.append(list(mac_morpho.sents(file)))
        file_list.append(file)
    zip_shuffler = list(zip(test_list, file_list))
    random.shuffle(zip_shuffler)
    test_list, file_list = zip(*zip_shuffler)
    return test_list, file_list


def get_train_data():
    train_list = []
    for file in gutenberg.fileids():
        train_list.append(list(gutenberg.sents(file)))
    return train_list