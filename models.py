import math
import numpy as np
from multiprocessing import Pool, cpu_count
from time import time
import warnings

#################################
############# TFIDF #############
#################################

class TFIDF:
    def __init__(self, corpus=[], tokenizer=None):
        self.corpus = []              # corpus
        self.corpus_size = 0          # size of the corpus
        self.voc = []                 # vocabulary 
        self.voc_size = 0             # size of the vocabulary
        self.doc_len = []             # length of the documents 
        self.avgdl = 0                # average length of the documents 
        self.doc_freqs = []           # frequencies of word for every doc
        self.idf = {}                 # idf
        self.tokenizer = tokenizer    # tokenizer
        self.nd = {}                  # word -> number of documents with word


        if len(corpus) > 0:

          if tokenizer:
              print("Tokenizing the corpus ...")
              x = time()
              self.corpus = self._tokenize_corpus(corpus)
              x = time() - x
              print(f"Done! Time Elapsed: {x}")
          else: 
            self.corpus = corpus

          print("Computing tfs and creating the vocabulary ...")
          x = time()
          self._initialize()
          x = time() - x  
          print(f"Done! Time Elapsed: {x}")

          print("Computing idfs ...")
          x = time()
          self._calc_idf()
          x = time() - x    
          print(f"Done! Time Elapsed: {x}")
        
        else:
          warnings.warn("corpus not provided, remember to insert it before computing the scores!") 
          
    def _initialize(self):
        nd = {}      # word -> number of documents with word
        num_doc = 0  # total length of the corpus in words
        self.corpus_size = 0
        self.doc_freqs = []
        self.doc_len = []
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.voc = list(nd.keys())
        self.voc_size = len(self.voc)
        self.avgdl = num_doc / self.corpus_size
        self.nd = nd

    def _tokenize_corpus(self, corpus):  
        pool = Pool(cpu_count())    # Parallelize tokenization between cpus
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self):
        """
        Return the log of inverse document frequency
        """
        for word, freq in self.nd.items():              
            idf = math.log(self.corpus_size) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        """
        Calculate tfidf between query and all docs
        """

        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (np.log(1 + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate tfidf scores between query and subset of all docs
        """

        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (np.log(1 + q_freq))

        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]
    

################################
############# BM25 #############
################################


class BM25:
    def __init__(self, corpus = [], tokenizer=None):
        self.corpus = []              # corpus
        self.corpus_size = 0          # size of the corpus
        self.voc = []                 # vocabulary 
        self.voc_size = 0             # size of the vocabulary
        self.doc_len = []             # length of the documents 
        self.avgdl = 0                # average length of the documents 
        self.doc_freqs = []           # frequencies of word for every doc
        self.idf = {}                 # idf
        self.tokenizer = tokenizer    # tokenizer
        self.nd = {}                  # word -> number of documents with word

        if len(corpus) > 1: 

          if tokenizer:
              print("Tokenizing the corpus ...")
              x = time()
              self.corpus = self._tokenize_corpus(corpus)
              x = time() - x
              print(f"Done! Time Elapsed: {x}")
          else: 
            self.corpus = corpus

          print("Computing tfs and creating the vocabulary ...")
          x = time()
          self._initialize()
          x = time() - x  
          print(f"Done! Time Elapsed: {x}")

          print("Computing idfs ...")
          x = time()
          self._calc_idf()
          x = time() - x    
          print(f"Done! Time Elapsed: {x}")

        else: 
          warnings.warn("Corpus not provided. Remember to insert it before computing the scores !")

        
    def _initialize(self):
        nd = {}      # word -> number of documents with word
        num_doc = 0  # total length of the corpus in words
        self.corpus_size = 0
        self.doc_freqs = []
        self.doc_len = []
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.voc = list(nd.keys())
        self.voc_size = len(self.voc)
        self.avgdl = num_doc / self.corpus_size
        self.nd = nd

    def _tokenize_corpus(self, corpus):  
        pool = Pool(cpu_count())    # Parallelize tokenization between cpus
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
########## BM25 Okapi ##########
################################


class BM25Okapi(BM25):
    """
    Original Formulation by Robertson et al. 
    """
    def __init__(self, corpus = [], tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.average_idf = 0
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        """
        Robertson-Sparck Jones IDF. It produces negative scores when df > /2
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in self.nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        Calculate bm25 scores for every document
        """
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """

        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
########## BM25 Lucene #########
################################


class BM25Lucene(BM25):
    """
    BM25 Lucene
    Actually Lucene use a document length compressed in a lossy manner 
    to a one byte value. It gets only 256 distinct document lengths,
    in order to pre-compute the value: k1 * (1 - b + b*(L_lossy / L_avg))
    """
    def __init__(self, corpus = [], tokenizer=None, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.average_idf = 0
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        """
        Calculates frequencies of terms in documents and in corpus.
        """
        idf_sum = 0
        for word, freq in self.nd.items():
            idf = math.log(1 + (self.corpus_size - freq + 0.5)/(freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
        self.average_idf = idf_sum / len(self.idf)

    def get_scores(self, query):
        """
        computes bm25 for every document
        """
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
########## BM25 ATARI ##########
################################


class BM25ATIRE(BM25):
    """
    ATIRE Variant 
    """
    def __init__(self, corpus = [], tokenizer=None, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.average_idf = 0
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        """
        Return the log of inverse document frequency
        """
        for word, freq in self.nd.items():              
            idf = math.log(self.corpus_size) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        """
        conmputes bm25 for every document
        The TF component is multiplied by k1+1
        """
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * ((self.k1 + 1)* q_freq /
                                                 (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()
      
    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
############ BM25L #############
################################


class BM25L(BM25):
    """
    BM25L more suited for long documents
    It penalizes less long docs
    """

    def __init__(self, corpus = [], tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        """
        Computes idf for each word
        """
        for word, freq in self.nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        """
        Computes bm25 for every document 
        """
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]



################################
############ BM25+ #############
################################


class BM25Plus(BM25):
    """
    BM25+ version
    """
    def __init__(self, corpus = [], tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        for word, freq in self.nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        """
        Computes bm25 for every document
        """
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
############ BM25Adtp ##########
################################


class BM25Adpt(BM25):
    """
    BM25 with adaptive K
    """

    def __init__(self, corpus = [], tokenizer = None,  b=0.75):
        # Algorithm specific parameters
        self.b = b
        super().__init__(corpus, tokenizer)

    def _dfr(self, q, r):
      if r == 0:
        return self.corpus_size
      elif r == 1:
        return self.nd.get(q) or 0
      else: 
        return np.sum(self.ctd - r + 0.5 > 0)
    
    def _G(self, q, r):
      return math.log(self._dfr(q, r+1) + 0.5, 2) - math.log(self._dfr(q, r) + 1, 2) - \
          math.log(((self.nd.get(q) or 0) / self.corpus_size) + 0.5, 2) + math.log(self.corpus_size + 1, 2)

    def _calc_k(self, q):
      """
      Computes k given a query term
      """

      # Computing information gains until G(q, r) > G(q, r+1)
      Gs = [self._G(q, 0), self._G(q, 1)]
      r=2
      while True:
        if self._G(q, r) > Gs[-1]:
          Gs.append(self._G(q, r))
          r+=1
        else:
          break  

      # If G(q, r) > G(q, r+1) happens for r = 1, the optimal value of k is undefined,
      # we set k = 0.001, as in Trotman et al.
      if len(Gs) == 2:
        return 0.001
      
      ks = np.arange(0.001, 2.5, 0.001)                # array of candidates k
      res_to_minimize = np.array([np.sum([
          (Gs[i] / Gs[1] - (k + 1) * i / (k + i))**2 
          for i in range(len(Gs))]
          ) for k in ks])

      return ks[np.argmin(res_to_minimize)]
        

    def get_scores(self, query):
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            self.ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            k = self._calc_k(q)
            score += (self._G(q, 1)) * (q_freq * (k + 1) /
                                              (k * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            self.ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            k = self._calc_k(q)
            score += (self._G(q, 1)) * (q_freq * (k + 1) /
                                              (k * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


################################
########### TFldpIDF ###########
################################


class TFIDFldp(BM25):
    def __init__(self, corpus = [],tokenizer = None,  b=0.75, delta = 1):
        # Algorithm specific parameters
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self):
        for word, freq in self.nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        query = self.tokenizer(query)

        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)

        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (1 + np.log(1 + np.log(
                q_freq / (1 - self.b + self.b * doc_len / self.avgdl) + self.delta
                )))
            
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        query = self.tokenizer(query)

        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (1 + np.log(1 + np.log(
                q_freq / (1 - self.b + self.b * doc_len / self.avgdl) + self.delta
                )))
        return score.tolist()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]
    

##########################################
############# RETRIEVER WRAPPER ##########
##########################################


import math
import numpy as np
from multiprocessing import Pool, cpu_count
from time import time
import warnings

class Retriever:
    def __init__(self, corpus = [], tokenizer=None, model=None, k1=1.5, b=0.75, epsilon=0.25, delta = 0.5):
        
        self.corpus = []              # corpus
        self.corpus_size = 0          # size of the corpus
        self.voc = []                 # vocabulary 
        self.voc_size = 0             # size of the vocabulary
        self.doc_len = []             # length of the documents 
        self.avgdl = 0                # average length of the documents 
        self.doc_freqs = []           # frequencies of word for every doc
        self.idf = {}                 # idf
        self.tokenizer = tokenizer    # tokenizer
        self.nd = {}                  # word -> number of documents with word
        self.model = model            # model to compute scores

        # Algotihm specific parameters
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.delta = delta

        if len(corpus) > 0:
          if tokenizer:
              print("Tokenizing the corpus ...")
              x = time()
              self.corpus = self._tokenize_corpus(corpus)
              x = time() - x
              print(f"Done! Time Elapsed: {x}")
          else: 
            self.corpus = corpus

          print("Computing tfs and creating the vocabulary ...")
          x = time()
          self._initialize()
          x = time() - x  
          print(f"Done! Time Elapsed: {x}")
        
        else:
          warnings.warn("corpus not provided, remember to insert it before computing the scores!") 

        if model: 
            if model not in ['TFIDF', 'BM25Okapi', 'BM25Lucene', 'BM25ATIRE', 'BM25Plus', 'BM25L', 'BM25Adpt', 'TFIDFldp']:
                raise Exception(f"'{model}'' is not a valid model name. Pass one of the following names throught the Retriever.switch_to(model) method: \n'TFIDF', 'BM25Okapi', 'BM25Lucene', 'BM25ATIRE', 'BM25Plus', 'BM25L', 'BM25Adpt', 'TFIDFldp ")
            
            else:
              if model != 'BM25Adpt' and len(self.corpus) > 0:
          
                print("Computing idfs ...")
                x = time()
                self._calc_idf()
                x = time() - x    
                print(f"Done! Time Elapsed: {x}")

        else:
            warnings.warn("model type not specified, remember to specify it before computing the scores!")


    def _initialize(self):
        nd = {}      # word -> number of documents with word
        num_doc = 0  # total length of the corpus in words
        self.corpus_size = 0
        self.doc_freqs = []
        self.doc_len = []
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.voc = list(nd.keys())
        self.voc_size = len(self.voc)
        self.avgdl = num_doc / self.corpus_size
        self.nd = nd

    def switch_to(self, model):
      if model not in ['TFIDF', 'BM25Okapi', 'BM25Lucene', 'BM25ATIRE', 'BM25Plus', 'BM25L', 'BM25Adpt', 'TFIDFldp']:
          raise Exception(f"'{model}'' is not a valid model name. Pass one of the following names throught the Retriever.switch_to(model) method: \n'TFIDF', 'BM25Okapi', 'BM25Lucene', 'BM25ATIRE', 'BM25Plus', 'BM25L', 'BM25Adpt', 'TFIDFldp ")
      else:
          self.model = model
          if model != 'BM25Adpt':
            
            self.idf = {}
            print("Recomputing idfs ...")
            x = time()
            self._calc_idf()
            x = time() - x    
            print(f"Done! Time Elapsed: {x}")

        
    def _tokenize_corpus(self, corpus):  
        pool = Pool(cpu_count())    # Parallelize tokenization between cpus
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self):
        
        if self.model in ["TFIDF", "BM25ATIRE"]:
          for word, freq in self.nd.items():              
              idf = math.log(self.corpus_size) - math.log(freq)
              self.idf[word] = idf

        elif self.model == "BM25Okapi":
          idf_sum = 0

          negative_idfs = []
          for word, freq in self.nd.items():
              idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
              self.idf[word] = idf
              idf_sum += idf
              if idf < 0:
                  negative_idfs.append(word)
          self.average_idf = idf_sum / len(self.idf)

          eps = self.epsilon * self.average_idf
          for word in negative_idfs:
              self.idf[word] = eps

        elif self.model == "BM25Lucene": 
          for word, freq in self.nd.items():
              idf = math.log(1 + (self.corpus_size - freq + 0.5)/(freq + 0.5))
              self.idf[word] = idf

        elif self.model == "BM25L":
          for word, freq in self.nd.items():
              idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
              self.idf[word] = idf

        elif self.model in ["BM25Plus", "TFIDFldp"]:
          for word, freq in self.nd.items():
              idf = math.log((self.corpus_size + 1) / freq)
              self.idf[word] = idf

        elif self.model == "BM25Adpt":
          self.idf = {}




    def _dfr(self, q, r):
      if r == 0:
        return self.corpus_size
      elif r == 1:
        return self.nd.get(q) or 0
      else: 
        return np.sum(self.ctd - r + 0.5 > 0)
    
    def _G(self, q, r):  
      return math.log(self._dfr(q, r+1) + 0.5, 2) - math.log(self._dfr(q, r) + 1, 2) - \
          math.log(((self.nd.get(q) or 0) / self.corpus_size) + 0.5, 2) + math.log(self.corpus_size + 1, 2)

    def _calc_k(self, q):

      # Computing information gains until G(q, r) > G(q, r+1)
      Gs = [self._G(q, 0), self._G(q, 1)]
      r=2
      while True:
        if self._G(q, r) > Gs[-1]:
          Gs.append(self._G(q, r))
          r+=1
        else:
          break  

      # If G(q, r) > G(q, r+1) happens for r = 1, the optimal value of k is undefined,
      # we set k = 0.001, as in Trotman et al.
      if len(Gs) == 2:
        return 0.001
      
      ks = np.arange(0.001, 2.5, 0.001)                # array of candidates k
      res_to_minimize = np.array([np.sum([
          (Gs[i] / Gs[1] - (k + 1) * i / (k + i))**2 
          for i in range(len(Gs))]
          ) for k in ks])

      return ks[np.argmin(res_to_minimize)]


    def get_scores(self, query):

        query = self.tokenizer(query)
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)

        if self.model == "TFIDF": 

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * (np.log(1 + q_freq))
          return score

        elif self.model == "BM25Okapi":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * (q_freq /
                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
          return score
        
        elif self.model == "BM25Lucene":

          score = np.zeros(self.corpus_size)
          doc_len = np.array(self.doc_len)
          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * (q_freq * 
                                                (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
          return score


        elif self.model == "BM25ATIRE":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * ((self.k1 + 1)* q_freq / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
          return score

        elif self.model == "BM25L":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
              score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                      (self.k1 + ctd + self.delta)
          return score

        elif self.model == "BM25Plus":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
          return score

        elif self.model == "BM25Adpt":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              self.ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
              k = self._calc_k(q)
              score += (self._G(q, 1)) * (q_freq * (k + 1) /
                                                (k * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
          return score

        elif self.model == "TFIDFldp":

          for q in query:
              q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
              score += (self.idf.get(q) or 0) * (1 + np.log(1 + np.log(
                  q_freq / (1 - self.b + self.b * doc_len / self.avgdl) + self.delta
                  )))
              
          return score
          


    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

