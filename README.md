# nlp-encoder-assembly

Using Phrase2Vec (a la Kavita Ganesan & Gensim), the objective is to build a small system for training word embedding towards the end of phrase embedding to create an encoder. Ideally this would become a small project aimed at developing a standard template for use in preprocesing any sort of Natural Language corpus, from prose manuscripts, to poems, to journalistic essays or any other.

- A very small dataset, consisting of William Shakespeare's "Twelfth Night", a basic story (romance & drama), is used for development.

- Regular Expressions are used to clean and tokenize the data per @underthesea's noted parser.

First of all, Mz. Ganesan appreciates stop words as a way of identifying phrases, not just parsing word or sentence tokens.

- I've chosen the Snowball stop word list as it is one of the oldest in use and published.

- Then, following Ganesan's process, the Phrase2Vec dataset of uni/bi/tri-grams will be passed through the Word2Vec algorithm. 

At this point, pre-processing would be finished, and this assembly would be complete.

- ...Usually, this pre-processing would be followed by an RNN-decoder, or some other NLP framework: attention, semantic or sentiment analysis, or what have you. 

- For this experiment, I will create a Variational AutoEncoder (VAE) that takes in input sequences that have been created from a Word2Vec style embedder. (I'm not yet sure about the semantic similarity between Phrase2Vec phrases versus PCA & tSNE embedding space similarities.)


