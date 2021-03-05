# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 01:17:46 2020

@author: ritu,nikita
"""
# importing all the required libraries
import math
from math import floor
import string
from string import digits
import nltk
import random as ran 
nltk.download('brown')
nltk.download('punkt')
from nltk.corpus import brown
import numpy as np
import matplotlib.pyplot as plt
import os


# making a class of unigram model
class unigram:
    SENTENCE_START = "<s>"
    SENTENCE_END = "</s>"
    
    def __init__(self, dataset):
         unigram_dictionary = dict()
         unigram_corpus_length = 0
         unigram_vocabulary = 0
         for sentence in dataset:
             for word in sentence:
                 if word in unigram_dictionary:
                     unigram_dictionary[word]=unigram_dictionary[word]+1
                 else:
                     unigram_dictionary[word]=1
                 if word != self.SENTENCE_START and word != self.SENTENCE_END:
                     unigram_corpus_length += 1
         unigram_vocabulary = len(unigram_dictionary) - 2
         self.unigram_dictionary = unigram_dictionary
         self.unigram_corpus_length = unigram_corpus_length
         self.unigram_vocabulary = unigram_vocabulary

    #return unigram dictionary
    def getUnigramDictionary(self):
        return self.unigram_dictionary
    
    #return unigram vocabulary
    def getUnigramVocabulary(self):
        return self.unigram_vocabulary
    
    #unigram word probability 
    def wordprobability(self, word,lambda_val = 0.0):
        word_probability_numerator = self.unigram_dictionary.get(word, 0) + lambda_val
        word_probability_denominator = self.unigram_corpus_length + (self.unigram_vocabulary)*lambda_val
        return float(word_probability_numerator) / float(word_probability_denominator)
    
    #unigram sentence probability 
    def unigram_sentence_probability(self, sentence,lambda_val = 0.0):
        sent_prob_log=0
        for word in sentence:
            if word != self.SENTENCE_START and word != self.SENTENCE_END:
                word_probability = self.wordprobability(word, lambda_val)
                sent_prob_log += math.log(word_probability,2)
        return sent_prob_log   

    # calculate unigram perplexity
    def calculate_unigram_training_perplexity(self, dataset,lambda_val = 0.0):
        unigram_word_count = 0
        sent_prob_log_sum = 0
        for sentence in dataset:
        # remove two for <s> and </s>
            unigram_word_count += (len(sentence) - 2)
        #retreive sentence probability
        #use log because the probabilities can be zero; sentence 8197 had zeroi
            try:
                sent_prob_log_sum -= self.unigram_sentence_probability(sentence,lambda_val)
            except:
                sent_prob_log_sum -= float('-inf')
        unigram_perplexity = math.pow(2, sent_prob_log_sum/unigram_word_count)
        return unigram_perplexity


# making a class of bigram model    
class bigram:
    SENTENCE_START = "<s>"
    SENTENCE_END = "</s>"
    
    def __init__(self, dataset):
        bigram_dict = dict()
        for sentence in dataset:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    if (previous_word, word) in bigram_dict:
                        bigram_dict[(previous_word, word)] = bigram_dict[(previous_word, word)]+1
                    else:
                        bigram_dict[(previous_word, word)]=1
                previous_word = word
        self.bigram_dict = bigram_dict
        
    
    #get bigram dictionary
    def getBigramDictionary(self):
        return self.bigram_dict

        
    # bigram word probability 
    def calculate_bigram_probabilty(self, previous_word, word, unigram_dictionary,lambda_val=0.0):
        bigramword_probability_numerator = self.bigram_dict.get((previous_word, word), 0) + lambda_val
        bigramword_probability_denominator = unigram_dictionary.get(previous_word, 0) + (len(unigram_dictionary)-2)*lambda_val
        if bigramword_probability_denominator == 0:
            return 0
        else: 
            return float(bigramword_probability_numerator) / float(bigramword_probability_denominator)


    # bigram sentence probability by not taking log
    def calculate_bigram_sentence_probability(self, sentence, unigram_dictionary, lambda_val=0.0):
        bigram_sentence_probability= 0
        previous_word = None        
        for word in sentence:
            if previous_word != None:
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word, unigram_dictionary, lambda_val)
                bigram_sentence_probability = bigram_sentence_probability + math.log(bigram_word_probability , 2)
            previous_word = word
        return bigram_sentence_probability 


    # calculate bigram perplexity
    def calculate_bigram_training_perplexity(self, dataset,uni_dict,lambda_val = 0.0):
        bigram_word_count = 0
        bigram_sent_prob_log_sum = 0
        for sentence in dataset:
        # remove <s> and </s>
            bigram_word_count += (len(sentence) - 1)
        #retreive sentence probability
            try: 
                bigram_sent_prob_log_sum -= self.calculate_bigram_sentence_probability(sentence,uni_dict,lambda_val)
            except:
                bigram_sent_prob_log_sum -= float('-inf')
        bigram_perplexity = math.pow(2, bigram_sent_prob_log_sum/bigram_word_count)
        return bigram_perplexity


#part 2e
#trigram
# making a class of trigram model    
class trigram:
    SENTENCE_START = "<s>"
    SENTENCE_END = "</s>"

    def __init__(self, dataset):
        trigram_dict = dict()
        sentence_count=0
        for sentence in dataset:
            sentence_count +=1
            previous_word = None
            prior_previous_word = None
            for word in sentence:
                if previous_word != None and prior_previous_word!= None:
                    if (prior_previous_word, previous_word, word) in trigram_dict:
                        trigram_dict[(prior_previous_word, previous_word, word)] = trigram_dict[(prior_previous_word, previous_word, word)]+1
                    else:
                        trigram_dict[(prior_previous_word, previous_word, word)]=1
                prior_previous_word = previous_word
                previous_word = word     
        self.trigram_dict = trigram_dict
        self.sentence_count = sentence_count
    

    #get trigram dictionary
    def getTrigramDictionary(self):
        return self.trigram_dict
    
    # trigram word probability 
    def calculate_trigram_probabilty(self, prior_prev_word ,previous_word, word, unigram_dictionary, bigram_dictionary,lambda_val=0.0):
        trigramword_probability_numerator = self.trigram_dict.get((prior_prev_word,previous_word, word), 0) + lambda_val
        
        if previous_word != self.SENTENCE_START:
            trigramword_probability_denominator = bigram_dictionary.get((prior_prev_word,previous_word), 0) + (len(unigram_dictionary)-2)*lambda_val
        else:
            trigramword_probability_denominator = self.sentence_count + (len(unigram_dictionary)-2)*lambda_val
            
        if trigramword_probability_denominator == 0:
            return 0
        else: 
            return float(trigramword_probability_numerator) / float(trigramword_probability_denominator)
    
    # trigram sentence probability by not taking log
    def calculate_trigram_sentence_probability(self, sentence, unigram_dictionary, bigram_dictionary, lambda_val=0.0):
        trigram_sentence_probability= 0
        prior_prev_word = None
        previous_word = None        
        for word in sentence:
            if previous_word != None and prior_prev_word != None:
                trigram_word_probability = self.calculate_trigram_probabilty(prior_prev_word, previous_word, word, unigram_dictionary, bigram_dictionary, lambda_val)
                trigram_sentence_probability = trigram_sentence_probability + math.log(trigram_word_probability,2) 
            prior_prev_word = previous_word
            previous_word = word
        return trigram_sentence_probability
    
    # calculate trigram perplexity
    def calculate_trigram_training_perplexity(self, dataset,uni_dict,bi_dict,lambda_val = 0.0):
        trigram_word_count = 0
        trigram_sent_prob_log_sum = 0
        for sentence in dataset:
        # remove <s> and </s>
            trigram_word_count += (len(sentence) - 2)
        #retreive sentence probability
            try: 
                trigram_sent_prob_log_sum -= self.calculate_trigram_sentence_probability(sentence,uni_dict, bi_dict,lambda_val)
            except:
                trigram_sent_prob_log_sum -= float('-inf')
        trigram_perplexity = math.pow(2, trigram_sent_prob_log_sum/trigram_word_count)
        return trigram_perplexity
    
    
#########################################################################################

# making a cleaner function to clean the input datasets for unigram and bigram model
#clean corpus function
def cleaner(dataset): 
    cleaned_data = []
    for sentences in dataset:
        x = " ".join(sentences)
        words = x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        # convert to lower case
        words = words.lower()
        # removing numbers
        words = words.translate(str.maketrans('','', digits))
        words_token = nltk.word_tokenize(words)
        words_token.insert(0,SENTENCE_START)
        words_token.append(SENTENCE_END)
        cleaned_data.append(words_token)
    return cleaned_data

#part e
#function to clean trigram data   
def trigram_cleaner(dataset):
    trigram_cleaned_data = []
    for sentences in dataset:
        x = " ".join(sentences)
        words = x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        # convert to lower case
        words = words.lower()
        # removing numbers
        words = words.translate(str.maketrans('','', digits))
        words_token = nltk.word_tokenize(words)
        words_token.insert(0,SENTENCE_START)
        words_token.insert(0,SENTENCE_START)
        words_token.append(SENTENCE_END)
        trigram_cleaned_data.append(words_token)
    return trigram_cleaned_data        
            

# calling main function    
if __name__ == '__main__':
    #split the data into training, validation and test
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    train_split_index = floor(len(brown.sents()) * train_split)
    validation_split_index = floor(len(brown.sents()) * validation_split)
    test_split_index = floor(len(brown.sents()) * test_split)
    
    train_set=brown.sents()[:train_split_index]
    validation_set=brown.sents()[train_split_index:train_split_index+validation_split_index]
    test_set=brown.sents()[train_split_index+validation_split_index:]
    
    #setting start and end of sentence variables
    SENTENCE_START = "<s>"
    SENTENCE_END = "</s>"
    
    # unigram model
    # modelling on training data set
    cleaned_train_data = cleaner(train_set)
    cleaned_validation_data = cleaner(validation_set)
    cleaned_test_data = cleaner(test_set)
    
    # training of model
    unigram_obj = unigram(cleaned_train_data)
    unigram_perplexity_train = unigram_obj.calculate_unigram_training_perplexity(cleaned_train_data)
    print("Train perplexity for unigram =" + str(unigram_perplexity_train))
    
    # testing of validation data
    unigram_perplexity_validation = unigram_obj.calculate_unigram_training_perplexity(cleaned_validation_data)
    print("Validation perplexity for unigram =" + str(unigram_perplexity_validation))
    
    # testing of test data
    unigram_perplexity_test = unigram_obj.calculate_unigram_training_perplexity(cleaned_test_data)
    print("Test perplexity for unigram =" + str(unigram_perplexity_test))
    
    
    # bigram model
    #making model on train data
    bigram_obj = bigram(cleaned_train_data)
    bigram_perplexity_train = bigram_obj.calculate_bigram_training_perplexity(
        cleaned_train_data,
        unigram_obj.getUnigramDictionary()
        )
    print("Train perplexity for bigram =" + str(bigram_perplexity_train))
    
    # testing on validation data
    bigram_perplexity_validation = bigram_obj.calculate_bigram_training_perplexity(
        cleaned_validation_data,
        unigram_obj.getUnigramDictionary()
        )
    print("Validation perplexity for bigram =" + str(bigram_perplexity_validation))
    
    
    # testing on test data
    bigram_perplexity_test = bigram_obj.calculate_bigram_training_perplexity(
        cleaned_test_data,
        unigram_obj.getUnigramDictionary()
        )
    print("Test perplexity for bigram =" + str(bigram_perplexity_test))
    
    
    #trigram model
    #part e - trigram implementation    
    trigram_cleaned_train_data = trigram_cleaner(train_set)
    trigram_cleaned_validation_data = trigram_cleaner(validation_set)
    trigram_cleaned_test_data = trigram_cleaner(test_set)
    
    trigram_obj = trigram(trigram_cleaned_train_data)
    
    
    trigram_perplexity_train = trigram_obj.calculate_trigram_training_perplexity(
        trigram_cleaned_train_data,
        unigram_obj.getUnigramDictionary(),
        bigram_obj.getBigramDictionary()
        )
    print("Train perplexity for trigram =" + str(trigram_perplexity_train))
    
    # testing on validation data
    trigram_perplexity_validation = trigram_obj.calculate_trigram_training_perplexity(
        trigram_cleaned_validation_data,
        unigram_obj.getUnigramDictionary(),
        bigram_obj.getBigramDictionary()
        )
    print("Validation perplexity for trigram =" + str(trigram_perplexity_validation))
    
    # testing on test data
    trigram_perplexity_test = trigram_obj.calculate_trigram_training_perplexity(
        trigram_cleaned_test_data,
        unigram_obj.getUnigramDictionary(),
        bigram_obj.getBigramDictionary(),
        )
    print("Test perplexity for trigram =" + str(trigram_perplexity_test))
    

################################################################################################
## part b
# unigram model on validation set
    unigram_lambda_list = np.arange(0.0001, 6, 0.01)
    unigram_perplexity_list = []
    for lambda_val in unigram_lambda_list:
        unigram_perplexity_validation = unigram_obj.calculate_unigram_training_perplexity(cleaned_validation_data, lambda_val)
        #look for the smallest perplexity
        unigram_perplexity_list.append(unigram_perplexity_validation)
   
    plt.plot(unigram_lambda_list, unigram_perplexity_list)
    plt.title('Unigram Perplexity')
    plt.xlabel('Lambda')
    plt.ylabel('Perplexity')
    plt.show()    
        
    unigram_smallest_perplexity = unigram_perplexity_list[0]
    unigram_smallest_lambda = unigram_lambda_list[0]
    for i in range(0,len(unigram_perplexity_list)-1):
        if(unigram_perplexity_list[i] < unigram_smallest_perplexity):
            unigram_smallest_perplexity = unigram_perplexity_list[i]
            unigram_smallest_index = i
    unigram_smallest_lambda = unigram_lambda_list[unigram_smallest_index]
    print("Smallest preplexity is %s at lambda %s" % (unigram_smallest_perplexity,unigram_smallest_lambda))
    #Smallest preplexity is 1580.6368975593302 at lambda 1.8601
      

# bigram model on validation set
    bigram_lambda_list = np.arange(0.0001, 1, 0.001)
    bigram_perplexity_list = []
    for lambda_val in bigram_lambda_list:
        bigram_perplexity_validation = bigram_obj.calculate_bigram_training_perplexity(cleaned_validation_data,unigram_obj.getUnigramDictionary(),lambda_val)
        bigram_perplexity_list.append(bigram_perplexity_validation)
       
    plt.plot(bigram_lambda_list, bigram_perplexity_list)
    plt.title('Biigram Perplexity')
    plt.xlabel('Lambda')
    plt.ylabel('Perplexity')
    plt.show()
    
    bigram_smallest_perplexity = bigram_perplexity_list[0]
    bigram_smallest_lambda = bigram_lambda_list[0]
    for i in range(0,len(bigram_perplexity_list)-1):
        if(bigram_perplexity_list[i] < bigram_smallest_perplexity):
            bigram_smallest_perplexity = bigram_perplexity_list[i]
            bigram_smallest_index = i
    bigram_smallest_lambda = bigram_lambda_list[bigram_smallest_index]
    print("Smallest preplexity is %s at lambda %s" % (bigram_smallest_perplexity,bigram_smallest_lambda))
    #Smallest preplexity is 1417.7725628100438 at lambda 0.0051
   

# trigram model on validation set
    trigram_lambda_list = np.arange(0.0001, 1, 0.001)
    trigram_perplexity_list = []
    for lambda_val in trigram_lambda_list:
        trigram_perplexity_validation = trigram_obj.calculate_trigram_training_perplexity(trigram_cleaned_validation_data, unigram_obj.getUnigramDictionary() ,bigram_obj.getBigramDictionary(),lambda_val)
        trigram_perplexity_list.append(trigram_perplexity_validation)
       
    plt.plot(trigram_lambda_list, trigram_perplexity_list)
    plt.title('Trigram Perplexity')
    plt.xlabel('Lambda')
    plt.ylabel('Perplexity')
    plt.show()
    
    trigram_smallest_perplexity = trigram_perplexity_list[0]
    trigram_smallest_lambda = trigram_lambda_list[0]
    trigram_smallest_index = 0
    for i in range(0,len(trigram_perplexity_list)-1):
        if(trigram_perplexity_list[i] < trigram_smallest_perplexity):
            trigram_smallest_perplexity = trigram_perplexity_list[i]
            trigram_smallest_index = i
    trigram_smallest_lambda = trigram_lambda_list[trigram_smallest_index]
    print("Smallest preplexity is %s at lambda %s" % (trigram_smallest_perplexity,trigram_smallest_lambda))
    #Smallest preplexity is 8216.692593833268 at lambda 0.0021
    
###############################################################################################

#part c
# train the unigram and bigram models for (training + validation) dataset

    # unigram model
    # merging training and validation dataset
    training_validation_dataset = cleaned_train_data + cleaned_validation_data
    # training of model 
    train_val_unigram_obj = unigram(training_validation_dataset)
    #training perplexity is not needed
    unigram_perplexity_train_val = train_val_unigram_obj.calculate_unigram_training_perplexity(training_validation_dataset, 1.8601)
    print("Train + Validation dataset perplexity for unigram =" + str(unigram_perplexity_train_val))
    
    # testing of test data
    test_partc_unigram_perplexity = train_val_unigram_obj.calculate_unigram_training_perplexity(cleaned_test_data,  1.8601)
    print("test perplexity for unigram from training and validation =" + str(test_partc_unigram_perplexity))
    
    
    # bigram model
    #making model on training + validation data
    #training perplexity is not needed
    train_val_bigram_obj = bigram(training_validation_dataset)
    bigram_perplexity_train_val = train_val_bigram_obj.calculate_bigram_training_perplexity(
        training_validation_dataset,
        train_val_unigram_obj.getUnigramDictionary(),
        0.0051
        )
    print("Train + Valdation dataset perplexity for bigram =" + str(bigram_perplexity_train_val))
    
    # testing on test data on model run for (training + validation) dataset
    test_partc_bigram_perplexity = train_val_bigram_obj.calculate_bigram_training_perplexity(
        cleaned_test_data,
        train_val_unigram_obj.getUnigramDictionary(),
        0.0051
        )
    print("Test perplexity for bigram from training and validation =" + str(test_partc_bigram_perplexity))
    
    #part (e)
    #trigram model
    # merging training and validation dataset for trigram model
    trigram_training_validation_dataset = trigram_cleaned_train_data + trigram_cleaned_validation_data
    # training of model
    trigram_train_val_obj = trigram(trigram_training_validation_dataset)
    #training perplexity is not needed
    trigram_perplexity_train_val = trigram_train_val_obj.calculate_trigram_training_perplexity(
        trigram_training_validation_dataset, train_val_unigram_obj.getUnigramDictionary() ,
        train_val_bigram_obj.getBigramDictionary(), 0.0021)

    print("Train + Validation dataset perplexity for trigram =" + str(trigram_perplexity_train_val))
    
    # testing of test data
    test_partc_trigram_perplexity = trigram_train_val_obj.calculate_trigram_training_perplexity(
    trigram_cleaned_test_data, train_val_unigram_obj.getUnigramDictionary() ,
        train_val_bigram_obj.getBigramDictionary(), 0.001)
    print("test perplexity for trigram from training and validation =" + str(test_partc_trigram_perplexity))
    
    
###############################################################################################
    
#part d
 
    ################# sentence generator for unigram model #################
    #since there is no context in unigram, we will be picking the words randomly from the unigram dictionary
    unigram_dict = train_val_unigram_obj.getUnigramDictionary()
    unigram_keys = unigram_dict.keys()
    list_unigram_keys = list(unigram_keys)
    
    #training_validation_dataset
    list_of_start = []
    for sentence in training_validation_dataset:
        start_words = []
        for k in range(0,len(sentence)):
            start_words.append(sentence[k])
        #since first and last word is currently <s> and </s>,
        #we will consider the word after <s> and before </s> as first and last word
        list_of_start.append(start_words[1])
    
    #generating 5 sentences
    randomsentence = [[] for i in range(0,5)]
    for i in range(0,5):        
        sindex = ran.randint(0,len(list_of_start)-1)
        start_word = list_of_start[sindex]
        randomsentence[i].append(list_of_start[sindex])
        word = start_word
        while(word != SENTENCE_END):
            nindex = ran.randint(0,len(list_unigram_keys)-1)
            word = list_unigram_keys[nindex]
            randomsentence[i].append(word)
        #print all the five randomly generated sentences
        print(" ".join(randomsentence[i]))     
      
    #printing the sentences in an output file   
    five_randomsentence=[" ".join(i) for i in randomsentence]
    path=r'C:\Users\ritu2\Desktop\UIC MSBA\Sem 2\Text Analytics\Assignments\Assignment 1'
    os.chdir(path)
    unigram_file = open("UnigramSentenceGeneration.txt","w") 
    count_sentence=1
    for i in five_randomsentence:  
        # \n is placed to indicate EOL (End of Line) 
        unigram_file.write("Start of Sentence "+str(count_sentence)+"\n\n") 
        unigram_file.writelines(str(five_randomsentence)) 
        unigram_file.write("\n\n" + "End of Sentence "+str(count_sentence)+"\n\n")   
        count_sentence=count_sentence+1
    unigram_file.close() #to change file access modes 

    ################# sentence generator for bigram model #################
    #trigram_cleaned_data = trigram_data(train_set)
    #trigram_obj = trigram(trigram_cleaned_data)
    #tri_dict = trigram_obj.getTrigramDictionary()    
    
    bigram_dict_train_val = train_val_bigram_obj.getBigramDictionary()
    bigram_keys_train_val = bigram_dict_train_val.keys()
    list_bigram_keys = list(bigram_keys_train_val)
    
    for i in range(0,5):
        sentenceform = []
        st=list_of_start[ran.randint(0,len(list_of_start)-1)]
       
        st1=[x for x in list_bigram_keys if x[0]==st]
        word1=st1[ran.randint(0,len(st1)-1)]        
        sentenceform.append(word1)
        flag = 0
        while(True):

            if(flag == 1):
                break
          
            bigram_word_2 = word1[1]
            
            if(bigram_word_2 == SENTENCE_END):
                sentenceform.append(".")
                break

            bigram_list=[x for x in list_bigram_keys if x[0]==bigram_word_2]
            
            if ( len(bigram_list) == 0 ):
                sentenceform.append(".")
                flag = 1
                break
            else:
                index_start = ran.randint(0,len(bigram_list)-1)
        
            sentenceform.append(bigram_list[index_start])
            word1 = bigram_list[index_start]
            
        print(" ".join([x[0] for x in sentenceform]) + "\n\n")
        
        
     ################# sentence generator for trigram model #################
        
    tri_dict = trigram_train_val_obj.getTrigramDictionary()  
    for i in range(0,5):
        trigram_sentence= []
        trigram_keys_train_val = tri_dict.keys()
        
        list_trigram_keys = list(trigram_keys_train_val)
       
        trigram_start_start_word = list_of_start[ran.randint(0,len(list_of_start)-1)]
        trigram_start=[x for x in list_trigram_keys if x[0]==trigram_start_start_word ]

        trigram_word = trigram_start[ran.randint(0,len(trigram_start)-1)]
        trigram_sentence.append(trigram_word)
        
        while(True):
            
            word = trigram_word[1]
            word2 = trigram_word[2]
            
            if(word2 == SENTENCE_END ):
                trigram_sentence.append(".")
                break
            
            trigram_list=[x for x in list_trigram_keys if x[0]==word and x[1]==word2 ]
    
            
            if ( len(trigram_list) == 0 ):
                trigram_sentence.append(".")
                break
            else:
                trigram_start_index = ran.randint(0,len(trigram_list)-1)

            trigram_sentence.append(trigram_list[trigram_start_index])
            trigram_word = trigram_list[trigram_start_index]
    
        print(" ".join([x[0] for x in trigram_sentence]) + "\n\n")

       
               
######################################################################################
    
    
    
