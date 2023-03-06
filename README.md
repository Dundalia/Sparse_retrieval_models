# Sparse retrieval models

In this repository we present several *bag-of-words* information retrieval models.
The repo comes with a tutorial that evaluates the models on the *WikiIR1k* [[1](#ref)] dataset, and on two datasets taken from the *BEIR* benchmark [[10](#ref)]:  *FiQA-2018* , and *Touché-2020*. 

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

In the library is even contained a single wrapper class `Retriever` that can retrieve documents given a query and specifying a technique. It has the advantage to store the tokenized corpus and the dictionary of the occurrencies of tokens for each document, namely `Retriever.nd`. 

Which technique is currently in use is stored as a string into the attribute `Retriever.model`. To change the model in use, it is necessary to run the methos `Retriever.switch_to(model_name)` and the `model_name` should be one between: 

`'TFIDF', 'BM25Okapi', 'BM25Lucene', 'BM25ATIRE', 'BM25Plus', 'BM25L', 'BM25Adpt', 'TFIDFldp'`. 

It will recumpute the idfs and store them in `Retriever.idf`.

### Overview

We took some of the models and the general class structure by the library [rank-bm25](https://github.com/dorianbrown/rank_bm25). 

All the models come with similar attributes: 

* `model.corpus`      = the corpus of documents           
* `model.corpus_size` = the number of documents in the corpus
* `model.voc`         = the vocabulary 
* `model.voc_size`    = the number of words in the vocabulary 
* `model.doc_len`     = the list of the length of the documents
* `model.avgdl`       = the average of the above list  
* `model.doc_freqs`   = a list of vocabularies containing the occurrencies of each word per document
* `model.idf`         = the vocabulary of idfs for each word
* `model.tokenizer`   = the preprocessing function to tokenize documents and queries
* `model.nd`          = the document frequency for each word

And similar methods: 

* `model.get_scores(query)` = returns the scores for each document
* `model.get_batch_scores(query, documents_ids)` = returns the scores for a subset of the corpus, indexed by documents_ids
* `model.get_top_n(query, n, documents)` = return the n documents with the highest scores


### Explanation of the models

#### $TFIDF$

The base model is **TF-IDF**. There are several versions of this model, that slighlty differs in how **TF** (term frequency) or **IDF** (inverse document frequency) are computed. Anyway the main idea is that TF accounts for the number of occurrencies od a term in a document, as a proxy for relevance, while IDF penalizes the most commoon terms in the corpus. It does so by scoring *rarity*:  If a term doesn’t occur in most documents in the corpus, then whenever it does occur, we’ll guess that this occurrence is significant. On the other hand, if a term occurs in most of the documents in our corpus, then the presence of that term in any particular document will lose its value as an indicator of relevance.

In our version of **TF-IDF**, we have computed TF and IDF as follows: 
> $$TF_{td} = \log(1 + tf_{td}), $$

Where $tf_{td}$ is the number of occurrencies of the term $t$ in doc $d$. And,  

> $$IDF_t = \log_{10}(N/df_t)$$

Where $df_t$ is the number of documents containing the term $t$. 

Finally we can compute the relevance score between a query $q$ and a document $d$ as: 

> $$TFIDF_{qd} = \sum_{t\in q}TF_{td} \cdot IDF_t$$


### $BM25$ Family

TF-IDF comes with several limitations and several improvements have been proposed. Nowdays, the family of **BM25** models represent the state of the art of bag-of-words retrieval methods.  BM25 is a ranking function based on the porbabilistic retrieval framework developed in the 1970s and 1980s by Stephen E. Robertson, Kaern Sparck Jones, and others. Let's see the limitations of TF-IDF and how BM25 solves those limitations (for an exhaustive explanation see [here](https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/)). 

**Term Saturation.** How should the ranking grow as the occurrencies of a term gorw in a specific document? Our TF-IDF implementation exploits the log of the occurrencies, so to make the ranking grow in logarithmic scale with respect to the term occurrencies. BM25 proposes another method: to control the saturation, BM25 uses the formula 

$$\frac{tf_{td}}{tf_{td} + k_1}.$$

Where $k_1$ is a constant that controls how rapidly the contribution approaches the upper bound of $TF$. Indeed the formula grows in a hyperbolic fashion and approaches $1$ as $tf_{td}$ goes to infinity. This formula comes with a fortunate side effect: if we have a query with more than one terms, let's say two, this ranking function has higher value in documents that contains one occurrency for each term, than 2 occurrencies of the same term.   

An interpretation of $k_1$ is that for documents of the average length, it is the value of the term frequency that gives a score of half the maximum score for the considered term.

**Document Length.** If a document happens to be really short and contains a particular keyword, it's a good indicator that that document is *about* that term, while we can't say the same for long documents. So we wuold like to penalize long documents. To achieve it, we adjust up $k_1$ if the document is longer than the average lenghth of the documents in the corpus, and adjust it down if the document is shorter than average, by multuplying $k_1$ by a factor that depends on the ratio $L_d / L_{avg}$, where $L_d$ is the document's length and $L_{avg}$ is the average document length across the corpus. 

To fine-tune the importance of the document length in the ranking we introduce another parameter $b$ (between $0$ and $1$), and multiply the $k_1$ by: 
$$1 - b + b \frac{L_d}{L_{avg}}.$$

**To sum up**, the final BM25 formula between a query $q$ and a document $d$ is: 
$$BM25_{q, d} = \sum_{t\in q} {IDF_t \cdot \frac{tf_{td}}{tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}}$$

Usually $k_1 \in [1.2, 2.0]$ and $b = 0.75.$

It happens that some variants of the $IDF$ function produce negative values. We don’t want negative values coming out of our ranking function because the presence of a query term in a document should never count against retrieval — it should never cause a lower score than if the term was simply absent. The variants of $BM25$ that I am goiing to present solve that issue in different ways. 

From that general idea, several variants have been porposed. We will implement some of them [[2](#ref)][[3](#ref)]: 

* $BM25 \ Okapi$
* $BM25 \ Lucene$
* $BM25 \ ATARI$
* $BM25L$
* $BM25+$
* $BM25 \ Adpt$
* $TF_{l\circ\delta\circ p}\times IDF$


### $BM25Okapi$

This is an adaptation of the original version [[3](#ref)], with slight improvements. 

Robertson et al. has proposed a variant of $IDF$, called $probabilistic \ IDF$. It is computed as: 
$$IDF_t = \log \left ( \frac{N - df_t + 0.5}{df_t + 0.5}\right )$$

To derive it teorethically we should go through the Robertson-Sparck Jones weights, but it goes beyond the scope of this project. Anyway to understand its behaviour we observe that it takes negative values approximately when the term appears in more than half the documents in the corpus. 

To deal with negative values, we set a lower bound to $IDFs$ to the average $IDF$ across all the terms in the vocabulary multiplyed by a constant $\epsilon$, that we set to $0.25.$

The final formula is: 

$$\sum_{t\in q} \log\left( \frac{N - df_t + 0.5}{df_t + 0.5} \right) \cdot \frac{tf_{td}} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$


### $BM25Lucene$ 

In the Lucene implementation, the only difference is how the negative values of $IDFs$ are managed: the $IDF$ formula is redefined as follows:

$$IDF_t = \log \left ( \frac{N - df_t + 0.5}{df_t + 0.5} +1 \right )$$

In this way, it can never take negative values.

The final formula is: 

$$\sum_{t\in q} \log \left ( \frac{N - df_t + 0.5}{df_t + 0.5} +1 \right ) \cdot \frac{tf_{td}} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$


### $BM25ATIRE$

In the ATIRE [[5](#ref)] implementation substitutes the probabilistic $IDF$ with the original definition of $IDF$: 

$$IDF_t = \log \left ( \frac{N}{df_t} \right ).$$

Even this implementation avoids negative values. 
The $TF$ component is multiplied by $(k_1 + 1)$ to make it look more like the Robertson-Sparck Jones weight, but it doesn't affect teh ranked list, as all scores are scaled linearly with this factor.  

The final formula is: 

$$\sum_{t\in q} \log \left ( \frac{N}{df_t} \right ) \cdot \frac{tf_{td} \cdot (1+k_1)} {tf_{td} + k_1 \cdot (1-b+b\cdot \frac{L_d}{L_{avg}})}$$

### $BM25L$

$BM25L$ [[6](#ref)] borns from the observation that $BM25$ penalizes longer documents too much compared to shorter ones.

 The $IDF$ component is defined as 
$$IDF_t = \log \left ( \frac{N+1}{df_t+0.5}\right )$$
to avoid negative values.

The $TF$ component is reformulated as $((k_1+1)\cdot c_{td})(k_1+c_{td})$ with $c_{td}= tf_{td} / (1-b+b\cdot (L_d/L_{avg}))$. It is further modified by adding a constant $\delta$, boosting the score for longer documents,  This has the effect of shifting the function to better favor small numbers (i.e. large denominators, equivalently large $L_d$ values, or long documents). The authors report $\delta = 0.5$ for highest effectiveness.

The final formula is:

$$\sum_{t\in q} \log \left ( \frac{N+1}{df_t+0.5}\right ) \cdot \frac{(k_1 +1)\cdot (c_{td}+\delta)} {k_1 + c_{td} + \delta}$$

### $BM25+$

Even this version [[7](#ref)] borns with the aim of dealing with the issue that ranking functions unfairly prefer shorter documents over longer ones. The proposal is to add a lower-bound bonus when a term appears at least one time in a document. The diﬀerence with $BM25L$ is a constant $\delta$ to the $TF$ component. The $IDF$ component is again changed to a variant that disallows negative values.

The final formula is: 

$$\sum_{t\in q} \log \left ( \frac{N+1}{df_t}\right ) \cdot \left( \frac{(k_1 +1)\cdot tf_{td}} {k_1 \cdot (1-b+b \cdot \frac{L_d}{L_{avg}}) + tf_{td}} + \delta \right )$$

### $BM25Adtp$

$BM25Adpt$ [[8](#ref)] is an approach that adapts $k_1$ for each term. To do so it starts by identifying the probability of a term occurring at least once in a document as $(df_{tr} + 0.5)/(N+1)$. The probability of the term occurring one more time is then deﬁned as $(df_{r+1} +0.5)/(df_r+ 1)$. The information gain of a term occurring $r + 1$ instead of $r$ times is deﬁned as 

$$G^r_q= \log_2 \left (\frac{df_{r+1} +0.5}{df_r +1}\right ) −\log_2\left ( \frac{df_{tr} +0.5}{N+1}\right),$$ 

where $df_r$ is deﬁned as follows: 

$$|\{D \ | \ c_{td} \ge r−0.5\}| \quad if \quad  r>1, \\ df_t \quad if \quad r=1,\\N \quad if \quad r=0$$

where $c_{td}$ is the same as in $BM25L$. That is to say, for the base case of $r = 0$, the number of documents in the collection is used; when $r = 1$, the document frequency is used; in all other cases, $|\{D \ | \ c_{td} \ge r−0.5\}|$, the number of documents, $|D|$, containing the term, $t$, that have a length normalized occurrence count, $c_{td}$, greater than $r$ (once rounded). The information gain is calculated for $r \in \{0,...,T\}$, until $G^r_q>G^{r+1}_q$. The optimal value for $k_1$ is then determined by ﬁnding the value for $k_1$ that minimizes the equation

$$k_1^* = argmin_{k_1}\sum_{r=0}^T \left ( \frac{G^r_q}{G^1_q}−\frac{(k_1+1)·r}{k_1+r} \right ) ^2.$$



Essentially, this gives avalue for $k_1 $that maximizes information gain for that speciﬁc term; $k_1^*$ and $G^1_q$ are then plugged into the $BM25Adpt$ formula, that is: 

$$\sum_{t\in q} G_q^1 \cdot \frac{(k_1^* +1) \cdot tf_{td}}{k_1^* \cdot (1-b+b\cdot(\frac{L_d}{L_{avg}})) + tf_{td}}$$

To compute the $k_1^*$ we have opted for a very simple solution: we try each value of $k_1 \in [0.001 , 2.5]$ with an increasing step of $0.001$ and take the one that minimizes the equation above. 

Kamphuis C. et al. [[3](#ref)] found that the optimal value of $k_1$ is actually not deﬁned for about $90 \%$ of the terms. A unique optimal value for $k_1$ only exists when $r>1$ while calculating $G^r_q$. For many terms, especially those with a low $df$ ,$G^r_q>G^{r+1}_q$ occurs before $r>1$. In these cases, picking different values for $k_1$ has virtually no effect on retrieval effectiveness. For undefined values, we set $k_1$ to $0.001$, the same as Trotman et al. [[2](#ref)].

### $TF_{l\circ\delta\circ p}\times IDF$

Rousseau & Vazirgiannis [[9](#ref)] suggest that nonlinear gain from
observing an additional occurrence of a term in a document
should be modeled using a log function thus: 

$$TF_{td} = 1 + \log(1 + \log(tf_{td}))$$

Following $BM25+$ they add $\delta$ to ensure there is a sufficient gap between the $0^{th}$ and $1^{st}$ term occurence, and than they apply the usual length normalization component of $BM25$. The $IDF$ function is the same as $BM25+$.

The final formula is: 

$$\sum_{t\in q} \log \left ( \frac{N+1}{df_t}\right ) \cdot  \left( 1+\log\left(1+\log\left(\frac{tf_{td}}{1-b+b\cdot \frac{L_d}{L_{avg}}} + \delta\right)\right)\right )$$



# References
<a name="ref"></a>

[1] Frej J., Schwab D., Chevallet J.P.,  WIKIR: A Python toolkit for building a large-scale Wikipedia-based English Information Retrieval Dataset, LREC 2020

[2] Trotman A., Puurula A., Burgess B., Improvements to BM25 and Language Models Examined, ADCS 2014

[3] Kamphuis C., P. de Vries A., Boytsov L., Lin J., Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants, Advances iin Information Retrieval 2020, p. 28-34 

[4] Robertson S., Walker S., Jones S., Hancock-Beaulieu M. M., Gatford M., OKapi at TREC-3, TREC-3 1995

[5] Trotman A., Jia X.F., Crane M., Towards an efficient and effective search engine, SIGIR 2012

[6] Lv Y., Zhai C., When documents are very long, BM25 fails! SIGIR, 2011

[7] Lv Y., Zhai C., Lower-bounding term frequency normalization, CIKM 2011

[8] Lv Y., C. Zhai, Adaptive term frequency normalization for
BM25, CIKM 2011, p. 1985-1988

[9] Rousseau F., M. Vazirgiannis, Composition of TF normalizations: new insights on scoring functions for ad hoc IR, SIGIR 2013

[10] Thakur N., Reimers N., Ruckle A., Srivastava A., Gurevych I., BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models, NeurIPS 2021








