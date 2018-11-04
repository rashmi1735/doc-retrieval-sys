
'''
    @Author : Rashmi Arora
    @Email  : rarora20@uic.edu
'''    

import os
import string
import math
import itertools
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from lxml import html
from bs4 import BeautifulSoup
from collections import defaultdict, OrderedDict
from functools import lru_cache

def read_files(path):    
    doc_dict = {}
    for filename in os.listdir(path):
            file_path = os.path.join(path,filename)
            if(os.path.isfile(file_path)):
                with open(file_path, "r") as file:
                    soup = BeautifulSoup(' '.join([line.strip() for line in file]), "html.parser")
                    response = [content.text.strip() for content in soup.findAll(['docno', 'title', 'text'])]
                    doc_dict[int(response[0])] = ' '.join(response[1:])                    
    return doc_dict
                
    
def read_queries(query_path):    
    with open(query_path, 'r') as query_file:
        query_dict = {}
        queries = query_file.readlines()
        query_dict = {i+1: queries[i].strip() for i in range(len(queries))}        
    return query_dict        


def read_relevant_doc(relevance_path):    
    relevant_docs = defaultdict(list)        
    with open(relevance_path, 'r') as relevance_file:
        for line in relevance_file:
            line = line.strip().split(' ')
            relevant_docs[int(line[0])].append(int(line[1]))
    return relevant_docs           


def get_stopwords(stopword_path):
    stopword_set = set([''])
    with open(stopword_path,'r') as file:
        for line in file:
            stopword_set.add(line.strip())    
    stopword_set.update(stopwords.words('english'))
    return stopword_set


def text_preprocessor(doc_dict):
    stopword_path = 'stopwords.txt'
    input_substring = string.punctuation + '0123456789'    
    remove_dict = dict.fromkeys(input_substring, ' ')
    table = str.maketrans(remove_dict)
    stemmer  = PorterStemmer()
    stem = lru_cache(maxsize=50000)(stemmer.stem) #cache for stemmed words to speed up pre-processing
    stopword_set = get_stopwords(stopword_path) #generate stopword list
    
    def preprocess(content):
        content = content.translate(table) #convert all punctutation and numbers to whitespace
        content = content.lower().split() #convert text to lowercase and tokenize
        content = [word.strip() for word in content if word not in stopword_set] #remove stopwords
        content = [stem(word) for word in content] #stem the tokens
        
        #remove any stopwords formed after stemming and remove words with length <= 2
        content = [word for word in content if word not in stopword_set and len(word) > 2]
        return content
    
    doc_dict.update({k: preprocess(v) for k,v in doc_dict.items()})   
    return doc_dict


def build_inverted_index(doc_dict):
    inverted_index = {}            # a dictionary of dictionaries    
    for doc in doc_dict:    
        for word in doc_dict[doc]:
            if word in inverted_index:
                if doc in inverted_index[word]:  #word exists, doc exists
                    inverted_index[word][doc] += 1                                    
                else:                            #word exists, doc does not exist, 
                    inverted_index[word]['df'] += 1 #increase doc frequency by 1
                    inverted_index[word][doc] = 1                
            else:                                #word does not exist
                inverted_index[word] = {}         
                inverted_index[word]['df'] = 1
                inverted_index[word][doc] = 1            
    return inverted_index


def get_doc_length(N, inverted_index):    
    doc_length = defaultdict(float)    
    for word in inverted_index:
        idf = (math.log10(N / (inverted_index[word]['df'])))
        for doc in inverted_index[word]:
            if doc != 'df':
                doc_length[doc] += math.pow(((inverted_index[word][doc])) * idf, 2)
    for doc in doc_length:
        doc_length[doc] = math.sqrt(doc_length[doc])        
    return doc_length


def get_word_freq(query_dict):
    for key in query_dict:
        word_list = query_dict[key]
        query_dict[key] = dict([word, word_list.count(word)] for word in word_list)
    return query_dict


def get_query_length(N, query_dict, inverted_index):
    query_length = defaultdict(float)
    for key in query_dict:
        for word in query_dict[key]:
            if word in inverted_index:                
                query_length[key] += math.pow(((query_dict[key][word]) * math.log10(N/inverted_index[word]['df'])), 2)
    for query in query_length:
        query_length[query] = math.sqrt(query_length[query])
    return query_length


def get_cos_similarity(N, query_dict, inverted_index, doc_length, query_length):
    cos_similarity = {}    
    for query in query_dict:
        cos_similarity[query] = defaultdict(float)
        for word in query_dict[query]:
            if word in inverted_index:            
                idf = math.log10(N/inverted_index[word]['df'])
                tfq = (query_dict[query][word])
                for doc in inverted_index[word]:
                    if doc != 'df':
                        cos_similarity[query][doc] += (tfq*idf*(inverted_index[word][doc])*idf)/((doc_length[doc]) * (query_length[query]))
        cos_similarity[query] = OrderedDict(sorted(cos_similarity[query].items(), key = lambda x: x[1], reverse = True))
    return cos_similarity
    

def get_result(cos_similarity):     # to fetch (q_id, doc_id) tuples
    result = [(query_id, doc_id) for query_id in cos_similarity for doc_id in cos_similarity[query_id]]
    return result


def get_relevant_retrieved(relevant_docs, top_n):
    relevant_retrieved = {}
    for query_id in relevant_docs:
        relevant_retrieved[query_id] = [[value for value in relevant_docs[query_id] if value in top_n[query_id]]]    
    return relevant_retrieved


def take(n, iterable):                         # extended itertool recipe from https://docs.python.org/3/library/itertools.html 
    return list(itertools.islice(iterable, n)) # return first n items of the iterable as a list


def get_metrics(n, relevant_docs, cos_similarity):
    top_n = dict([(query_id, take(n, cos_similarity[query_id])) for query_id in cos_similarity]) 	# Top n retrieved
    relevant_retrieved = get_relevant_retrieved(relevant_docs, top_n) 			# Relevant ones from the top n retrieved for each query
    for q_id in relevant_retrieved:
        relevant_retrieved[q_id].append(len(relevant_retrieved[q_id][0])/ n)		# Precision
        relevant_retrieved[q_id].append(len(relevant_retrieved[q_id][0])/ len(relevant_docs[q_id])) 	# Recall   
    return relevant_retrieved


def get_avg(num_list):
    return sum(num_list) / len(num_list)    


if __name__ == "__main__":          
    path = ''    
    query_path = ''
    relevance_path = ''
    
    while True:
        path = input("Enter path to the directory where the document(s) are stored:\n")
        if(os.path.isdir(path)):
            break
        else:
            print("Invalid Path or No files exist\n")
    
    while True:
        query_path = input("Enter path to query file:\n")
        if(os.path.isfile(query_path)):
            break
        else:
            print("Invalid Path or Not a file\n")
    
    while True:
        relevance_path = input("Enter path to the relevance file:\n")
        if(os.path.isfile(relevance_path)):
            break
        else:
            print("Invalid Path or Not a file\n")
              
    print("\nFetching Results...")
    
    given_relevant_docs = read_relevant_doc(relevance_path) #store relevant docs from given text file    
    docs = read_files(path) #read all the documents    
    N = len(docs) #number of all documents    
    processed_docs = text_preprocessor(docs) #pre-process the documents    
    inverted_index = build_inverted_index(processed_docs)  #build an inverted index - Pass 1    
    doc_lengths = get_doc_length(N, inverted_index)  #calculate doc lengths - Pass 2    
    queries = read_queries(query_path) #read all the queries    
    processed_q = text_preprocessor(queries)  #pre-process the queries    
    query_word_freq = get_word_freq(processed_q) #get word frequency for each query    
    q_lengths = get_query_length(N, query_word_freq, inverted_index) #calculate query lengths
    
    #calculate cosine similarity between each query and the docs in which query terms occur
    cos_similarity = get_cos_similarity(N, query_word_freq, inverted_index, doc_lengths, q_lengths) 
    
    #store all retrieved documents in (query_id, doc_id) tuple format
    result = get_result(cos_similarity)
    
    #calculating metrics for top10, top50, top100 and top500 retrieved documents
    recall_levels = [10, 50, 100, 500]
    
    #store top n retrieved docs, precision and recall for each query for nth recall level 
    metrics_dict = {n : get_metrics(n, given_relevant_docs, cos_similarity) for n in recall_levels}
    
    #calculate average precision
    avg_p = {item : get_avg([metrics_dict[item][key][1] for key in metrics_dict[item]]) for item in metrics_dict}
    
    #calculate average recall
    avg_r = {item : get_avg([metrics_dict[item][key][2] for key in metrics_dict[item]]) for item in metrics_dict}
    
    #print results
    print("\n")    
    print("********************* RESULTS *******************")
    print("\n") 
    for n in metrics_dict:
        print("Top " + str(n) + " documents in rank list:")
        print("\t\tPrecision\tRecall")
        for i in metrics_dict[n]:
            print("Query: " + str(i) + "\t" + str(metrics_dict[n][i][1]) + "\t\t" + str(metrics_dict[n][i][2]))
        print("Average Precision: " + str(avg_p[n]))
        print("Average Recall: " + str(avg_r[n]))


        print("\n")    
    print("****************** END OF RESULTS ****************")    

