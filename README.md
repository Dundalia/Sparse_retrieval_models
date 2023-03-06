# Sparse retrieval models

In this notebook we are going to present several *bag-of-words* information retrieval models and test them on the *WikiIR1k* [[1](#ref)] dataset, and on two datasets taken from the *BEIR* benchmark [[10](#ref)]:  *FiQA-2018* , and *Touché-2020*. 

Several bag-of-words models have been developed for the task of information retrieval (IR). It consists in, given a set of documents and a query (a set of terms), find the most relevant documents for that query. Bag-of-words models make a simple but helpful assumption: the more a document contains a term, the more likely it is to be *about*  that term.

It is necessary to clarify the terminology and give some definitions: 
* $d$ - a document
* $q$ - a query
* $t$ - a term
* $N$ - size of the corpus of documents
* $df_t$ - document frequency of the term $t$, i.e. the number of documents in wich the term $t$ appears
* $tf_{td}$ - term-frequency between the term $t$ and the document $d$, i.e. the number of occurrencies of the term $t$ in the document $d$.
* $L_d$ - Length of the document $d$
* $L_{avg}$ - average length of the documents in the corpus

***

The different ranking functions that we are going to evaluate is listed in the following table:

|MODEL NAME | FORMULA| 
|--- | --- |
$$TFIDF$$ |  $$\sum_{t\in q} \log_{10}(N/df_t) \cdot \log(1 + tf_{td})$$
$BM25 \ Okapi$ | $$\sum_{t\in q} \log\left( \frac{N - df_t + 0.5}{df_t + 0.5} \right) \cdot \frac{tf_{td}} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$
$$BM25 \ Lucene$$ | $$\sum_{t\in q} \log \left ( \frac{N - df_t + 0.5}{df_t + 0.5} +1 \right ) \cdot \frac{tf_{td}} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$
$$BM25 \ ATIRE$$ | $$\sum_{t\in q} \log \left ( \frac{N}{df_t} \right ) \cdot \frac{tf_{td} \cdot (1+k_1)} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$
$$BM25L$$| $$\sum_{t\in q} \log \left ( \frac{N+1}{df_t+0.5}\right ) \cdot \frac{(k_1 +1)\cdot (c_{td}+\delta)} {k_1 + c_{td} + \delta}$$
$$BM25+$$ | $$\sum_{t\in q} \log \left ( \frac{N+1}{df_t}\right ) \cdot \left( \frac{(k_1 +1)\cdot tf_{td}} {k_1 \cdot (1-b+b \cdot \frac{L_d}{L_{avg}}) + tf_{td}} + \delta \right )$$
$$BM25\ Adtp$$ | $$\sum_{t\in q} G_q^1 \cdot \frac{(k_1^* +1) \cdot tf_{td}}{k_1^* \cdot (1-b+b\cdot(\frac{L_d}{L_{avg}})) + tf_{td}}$$
$$TF_{l\circ\delta\circ p}\times IDF$$ | $$\sum_{t\in q} \log \left ( \frac{N+1}{df_t}\right ) \cdot  \left( 1+\log\left(1+\log\left(\frac{tf_{td}}{1-b+b\cdot \frac{L_d}{L_{avg}}} + \delta\right)\right)\right )$$

The implementation is an extension and adaptation of the [rank-bm25](https://github.com/dorianbrown/rank_bm25) library.

The library comes with the implementations of $Okapi \ BM25$, $BM25L$, and $BM25+$. We have slightly improved them and added the ones in the table above. Furthermore we have implemented a wrapper `Retriever` class able to retrieve with any of the techniques previously listed.



