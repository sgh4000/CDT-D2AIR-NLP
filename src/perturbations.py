import pandas as pd
import numpy as np
import replicate
import time
import nltk
import csv
import re
import os
from sentence_transformers import SentenceTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from random import randint, seed
import random
#set seeds for reprodicibility
random.seed(42)
np.random.seed(42)


def create_perturbations(Xptn, Yptn, Xntn, Yntn, Xpts, Ypts, Xnts, Ynts, perturbation):
    if perturbation == 'character':
        perturbations = [char_swapping, char_replacement, char_deletion, char_insertion, char_repetition]
    seed(42)
    if perturbation == 'character' or perturbation == 'word':
        # Train positive
        X_train_pos_perturbed = []
        y_train_pos_perturbed = []
        train_pos_index = []
        for i in range(len(Xptn)):
            for perturbation in perturbations:
                p_perturbed = perturbation([Xptn[i]])
                X_train_pos_perturbed.append(p_perturbed[0])
                y_train_pos_perturbed.append(Yptn[i])
                train_pos_index.append(i)
        
        # Train negative
        X_train_neg_perturbed = []
        y_train_neg_perturbed = []
        train_neg_index = []
        for i in range(len(Xntn)):
            for perturbation in perturbations:
                p_perturbed = perturbation([Xntn[i]])
                X_train_neg_perturbed.append(p_perturbed[0])
                y_train_neg_perturbed.append(Yntn[i])
                train_neg_index.append(i)

        # Test positive
        X_test_pos_perturbed = []
        y_test_pos_perturbed = []
        test_pos_index = []
        for i in range(len(Xpts)):
            for perturbation in perturbations:
                p_perturbed = perturbation([Xpts[i]])
                X_test_pos_perturbed.append(p_perturbed[0])
                y_test_pos_perturbed.append(Ypts[i])
                test_pos_index.append(i)

        # Test negative
        X_test_neg_perturbed = []
        y_test_neg_perturbed = []
        test_neg_index = []
        for i in range(len(Xnts)):
            for perturbation in perturbations:
                p_perturbed = perturbation([Xnts[i]])
                X_test_neg_perturbed.append(p_perturbed[0])
                y_test_neg_perturbed.append(Ynts[i])
                test_neg_index.append(i)
   
    return X_train_pos_perturbed, y_train_pos_perturbed,  train_pos_index, X_train_neg_perturbed, y_train_neg_perturbed, train_neg_index, X_test_pos_perturbed, y_test_pos_perturbed, test_pos_index, X_test_neg_perturbed, y_test_neg_perturbed, test_neg_index

# Character perturbations
def return_random_number(begin, end):
    return randint(begin, end)


def return_adjacent_char(input_char):
    
    if (input_char == 'a'):
        return 's'
    
    elif (input_char == 'b'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'v'
        else:
            return 'n'
        
    elif (input_char == 'c'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'x'
        else:
            return 'v'
        
    elif (input_char == 'd'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 's'
        else:
            return 'f'
        
    elif (input_char == 'e'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'w'
        else:
            return 'r'
        
    elif (input_char == 'f'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'd'
        else:
            return 'g'
        
    elif (input_char == 'g'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'f'
        else:
            return 'h'
        
    elif (input_char == 'h'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'g'
        else:
            return 'j'
        
    elif (input_char == 'i'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'u'
        else:
            return 'o'
        
    elif (input_char == 'j'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'h'
        else:
            return 'k'
        
    elif (input_char == 'k'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'j'
        else:
            return 'l'
    
    elif (input_char == 'l'):
        return 'k'
        
    elif (input_char == 'm'):
        return 'n'
        
    elif (input_char == 'n'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'b'
        else:
            return 'm'
        
    elif (input_char == 'o'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'i'
        else:
            return 'p'
        
    elif (input_char == 'p'):
        return 'o'
    
    elif (input_char == 'q'):
        return 'w'
        
    elif (input_char == 'r'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'e'
        else:
            return 't'
        
    elif (input_char == 's'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'a'
        else:
            return 'd'
        
    elif (input_char == 't'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'r'
        else:
            return 'y'
        
    elif (input_char == 'u'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'y'
        else:
            return 'i'
    
    elif (input_char == 'v'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'c'
        else:
            return 'b'
        
    elif (input_char == 'w'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'q'
        else:
            return 'e'
        
    elif (input_char == 'x'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'z'
        else:
            return 'c'
        
    elif (input_char == 'y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 't'
        else:
            return 'u'
        
    elif (input_char == 'z'):
        return 'x'
    #---------------------------------------------
    elif (input_char == 'A'):
        return 'S'
    
    elif (input_char == 'B'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'V'
        else:
            return 'N'
        
    elif (input_char == 'C'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'X'
        else:
            return 'V'
        
    elif (input_char == 'D'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'S'
        else:
            return 'F'
        
    elif (input_char == 'E'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'W'
        else:
            return 'R'
        
    elif (input_char == 'F'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'D'
        else:
            return 'G'
        
    elif (input_char == 'G'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'F'
        else:
            return 'H'
        
    elif (input_char == 'H'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'G'
        else:
            return 'J'
        
    elif (input_char == 'I'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'U'
        else:
            return 'O'
        
    elif (input_char == 'J'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'H'
        else:
            return 'K'
        
    elif (input_char == 'K'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'J'
        else:
            return 'L'
    
    elif (input_char == 'L'):
        return 'K'
        
    elif (input_char == 'M'):
        return 'N'
        
    elif (input_char == 'N'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'B'
        else:
            return 'M'
        
    elif (input_char == 'O'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'I'
        else:
            return 'P'
        
    elif (input_char == 'P'):
        return 'O'
    
    elif (input_char == 'Q'):
        return 'W'
        
    elif (input_char == 'R'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'E'
        else:
            return 'T'
        
    elif (input_char == 'S'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'A'
        else:
            return 'D'
        
    elif (input_char == 'T'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'R'
        else:
            return 'Y'
        
    elif (input_char == 'U'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Y'
        else:
            return 'I'
    
    elif (input_char == 'V'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'C'
        else:
            return 'B'
        
    elif (input_char == 'W'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Q'
        else:
            return 'E'
        
    elif (input_char == 'X'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Z'
        else:
            return 'C'
        
    elif (input_char == 'Y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'T'
        else:
            return 'U'
        
    elif (input_char == 'Z'):
        return 'X'
    
    else:
        return '*'


def swap_characters(input_word, position, adjacent):
    temp_word = ''
    if (adjacent == 'left'):
        if (position == 1):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
        elif (position > 1 and position < len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
            temp_word += input_word[position+1:]
            
    elif (adjacent == 'right'):
        if (position == 0):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
        elif (position > 0 and position < len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
            temp_word += input_word[position+2:]
            
    return temp_word


def char_replacement(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if has_3_letter_word == False:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True

        #print('Sentence: ', sample_text)
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        char_is_letter = False
        tries_number = 0
        
        while (char_is_letter != True and tries_number <= 20):
            random_char_index = return_random_number(1, len(selected_word)-2)
            tries_number += 1
            if ((ord(selected_word[random_char_index]) >= 97 and ord(selected_word[random_char_index]) <= 122) or (ord(selected_word[random_char_index]) >= 65 and ord(selected_word[random_char_index]) <= 90)):
                char_is_letter = True
                is_sample_perturbed = True
        
        #print('Random position:', random_char_index)
        #print('Character to replace:', selected_word[random_char_index])
        
        #--------------------------- replace the character
    
        char_to_replace = selected_word[random_char_index]
        adjacent_char = return_adjacent_char(char_to_replace)
        #print('Adjacent character:', adjacent_char)
        
        temp_word = selected_word[:random_char_index]
        temp_word += adjacent_char
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After replacement:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)
    return np.array(new_sentences)
    

def char_swapping(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True

        #print('Sentence: ', sample_text)
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        random_char_index = return_random_number(0, len(selected_word)-1)

        #print('Random position:', random_char_index)
        #print('Char in random position:', selected_word[random_char_index])
        
        #--------------------------- select an adjacent for swapping
            
        adjacent_for_swapping = ''
        
        if (random_char_index == 0):
            adjacent_for_swapping = 'right'
        elif (random_char_index == len(selected_word)-1):
            adjacent_for_swapping = 'left'
        else:
            adjacent = return_random_number(1, 2)
            if(adjacent == 1):
                adjacent_for_swapping = 'left'
            else:
                adjacent_for_swapping = 'right'
                
        #print('Adjacent for swapping:', adjacent_for_swapping)

        #--------------------------- swap the character and the adjacent
            
        temp_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)
        perturbed_word = ""

        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After swapping:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_deletion(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        #print('Character to delete:', selected_word[random_char_index])
        
        #--------------------------- delete the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After deletion:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_insertion(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        
        #--------------------------- select a random character
        
        random_char_code = return_random_number(97, 122)
        #print('Random character:', chr(random_char_code))
    
        temp_word = selected_word[:random_char_index]
        temp_word += chr(random_char_code)
        temp_word += selected_word[random_char_index:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After insertion:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_repetition(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        #print('Character to repeat:', selected_word[random_char_index])
        
        #--------------------------- repeat the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index] + selected_word[random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After repetition:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)
