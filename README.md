# sentiment-analysis-RT

This repository contains Tensorflow code for a deep neural network to perform sentiment analysis on the [Rotten Tomatoes Movie Review dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

The dataset contains 1000 positive and 1000 negative movie reviews. The LSTM based Recurrent Neural Network trains on this dataset to learn how to classify positive and negative reviews / statements.

## Setup

- `tar -xvzf review_polarity.tar.gz` , which untars the RT dataset
- `tar -xvzf data.tar.gz`, which untars the GloVe numpy array and precomputed IDs array
- `tar -xvzf models.tar.gz`, which untars the model pretrained on a GPU machine


## Explanation

- `jupyter notebook` which launches the Sentiment-Analysis-Rotten-Tomatoes notebook with explained code.

## Run

- `python preprocess.py`, for computing the IDs [optional, in case data.tar.gz is not untared]
- `python train.py` for training on the dataset [option, in case models.tar.gz is not untared]
- `python test.py` for testing the model

## References

- https://nlp.stanford.edu/projects/glove/
- https://www.damienpontifex.com/2017/10/27/using-pre-trained-glove-embeddings-in-tensorflow/
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://www.oreilly.com/topics/ai
