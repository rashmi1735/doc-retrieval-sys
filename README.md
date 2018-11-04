# doc-retrieval-sys
Vector Space Model based Document Retrieval System

1) The program starts by taking three inputs from the user:
	1 - the path to the directory where all the documents are stored (Parent Directory only of all the documents)
	2 - the path to the queries file (Parent Directory + Filename)
	3 - the path to the relevance file (Parent Directory + Filename)
An absolute path is required for each of these files.

2) read_files():
- reads all the documents line by line and keeps content from only <docno>, <title> and <text> tags. BeautifulSoup library with html.parser has been used for this. Saves the documents in a dictionary: doc_dict - with docno as key and title and text together as value.

3) read_queries():
- reads queries from quereis.txt and stores as a dictionary with query number as key and query text as value

4) read_relevant_doc():
- reads the relevance.txt file and stores it as a dictionary with query number as key and list of corresponding relevant documents as value

5) get_stopwords():
- fetches a set of unique stopwords obtained after combining the stopwords from stopwords.txt and nltk stopwords

6) text_preprocessor():
- receives an unprocessed dictionary with doc_id/query_id as key and doc_text/query_text as value 
- creates a dictionary with punctuation and numbers in string format as keys and a whitespace ' ' as value for each of these keys.
- converts all punctuation and numbers to whitespace using this dictionary
- converts all text to lowercase
- splits the text on whitespace to form tokens
- removes stopwords before stemming
- performs stemming
- removes any stopwords formed after stemming
- removes all words with length <=2
- returns an updated dictionary with doc_id/query_id as key and preprocessed token list as value 

7) build_inverted_index():
- scans the pre-processed document dictionary word by word and builds an inverted index based on the logic discussed in class. The data structure used to store it is a dictionary of dictionaries

8) get_doc_length():
- this is the second pass through the inverted index to calculate the doc lengths. The doc lengths are computed incrementally as documents are encountered for every word in the inverted index. A dictionary is used with doc_id as key and doc_length as value

9) get_word_freq():
- builds a dictonary of dictionaries with query id as key and a dictionary of all unique terms(keys) in the corresponding query with their frequencies(values) as values

10) get_query_length():
- determines the query length and stores it in dictionary with query_id as key and query_length as value. Since the structure of doc_dict is different from that of query_dict, a separate function is used

11) get_cos_similarity():
- computes the cosine similarity between each query and the documents containing the query terms. Again a dictionary of dictionaries is used with query_id as key and a dictionary of all documents(keys) with corresponding cosine similarity(value) as value
- returns the dictionary with cosine similarity in sorted(descending) order

12) get_result():
- fetchs all (q_id, doc_id tuples) from the cosine similarity dictionary

13) get_relevant_retrieved():
- compares the given relevant documents and retrieved documents and returns the relevant documents from the retrieved ones for all queries. A dictionary is used - query id as key and list of relevant retrieved docs as value

14) take():
- takes an iterable as parameter and returns first n items of the iterable as a list

15) get_metrics():
- computes the precision and recall corresponding to each query using the relevant retrieved docs
- returns a list with list of relevant documents, precison and recall corresponding to each query and top n retrieved docs

16) get_avg():
- computes the average of values passed to it



INSTRUCTIONS ON HOW TO RUN THE CODE:

Requirements : Python3, nltk

1) Keep the InvertedIndex.py and stopwords.txt files in the same directory
2) Open the command line
3) Go to the directory where InvertedIndex.py and stopwords.txt files are stored
4) Run the below command

	python3 InvertedIndex.py

5) The program will ask you for 3 inputs:
	1 - the path to the directory where all the documents are stored (Parent Directory only of all the documents)
	2 - the path to the queries file (Parent Directory + Filename)
	3 - the path to the relevance file (Parent Directory + Filename)
Please provide an absolute path for all three. Press Enter after inputting each path.
6) The results for precision and recall for each query, average precision and average recall for top 10, top 50, top 100 and top 500 retrieved documents will be displayed.



