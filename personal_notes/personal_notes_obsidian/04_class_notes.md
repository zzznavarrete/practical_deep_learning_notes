After learn how a model can be fitted into some data, through an adjustment of its parameters through an minimization of the loss function.

Now we could put ourselves in the situation in which we already know some parameters (more or less) values, that would allow us to save time and computation. That is what a pre trained model is.
  
Fine tuning is the process of get aright the parameters of what we less know which values should took and optimize it and move the other ones (the ones that we already know more or less what values should takes) a little bit.

  

### Transformers
Transformers are actually pretty good at taking advantages of some modern TPU architecture (google TPU's exactly).

Transformers didn't really predict the next word of the sentence. Instead, they took kind of chunks of any text that it's beign train on, delete random words, and ask the model to predict which/wat words were deleted.

One of the most common application of transformers is classification. (ie, document classification) be aware, a document could be 3 o 4 word or an entiner encyclopedia, it's just and input of a transformer neural network.

But there are other applications such as:

- Sentiment analysis,
- Author identification,
- ec.

  
The detail here is that neural networks works with numbers, so there are basically 2 steps that are needed to take for NLP models.
1. Split into tokens: Basically into words.
2. Numericatlization of tokens.
  
The process of split each text up into words it's called 'Tokenization', and the process of convert each word (or token) into a number it's called 'Numericalization'

Before start tokenizing, it's needed to choose what model to use.
Hugging Face model hub has many many models which we could use.

Each of this architecture could be trained to different problems to apply to differents contexts.
A tokenized dataset will have an unique ID for each set of possible words. That way is how a string is converted into a list of numbers (numericalization).

  
  
### A powerful idea in machine learning: Train, validation and test set.

If for instance we wanted to fit a polynomial regression into a quadratic form random points, we'll see that, at increasing the polynomial degree, the fitted line will start to get more and more 'adjusted' very closely to the points (missing generalization), this problem is called 'Overfitting'. In the other hand, if we adjust a linear polynomial (degree 1), surely it will throw a line that cross the points without beign near close of the data shape, this is called 'Underfit'.

We don't want overfitting or underfitting, How do we reach to certain point in which we are in the middle point of underfitting/overfitting?

By default, we will fit the line/function with a portion of the data with removed points, then we'll measured the function only in the points that we previously removed, this dataset is called 'validation set'.
  
Fast ai dont let train model without validation sets.

  

### Creating a good validation set

Is not necessarily simple as you may think. If in a certain data, we removed only randomly data points is a poor choise, it's simple to fill the gaps and not indicativa of what we're goin to need in production.
Insted, if we truncate and remove a certain percentage of the most recent data for validation set, is more representative and will measure better the model performance.
Alhotugh scikit-learn offers a ```train_test_split``` method for building a train/validation set, this is a poor choise given that it took random data samples.


### Test set and what it is for
A way of know if we actually have a good model or not is to held back another dataset in which, after we complete the whole process of training/evaluating, try different models, etc, we must perform an evaluation of our model with this dataset. There could be a chance in which we perform good in our model and validation set just for coincidence, and in reality, we could be in a position of overfit, a way of know if we are in overfit or not it's actually having a ```test set``` to validate the model with.



### Metrics and correlation
What we're going to do with the validation set is getting metrics. A number that will tell us how good is the model.

Question. Is this the same metric that our loss function? (that was used to get the gradient and optimize the model parameters). The answer is, not necessarily.

Also in real life, the model that we choose, it's a path of a complex process, often involving humans, costumers, things changing over time, etc etc. One metric is not enough to capture all of that. Unfortunately, because is so convenient to pick only 1 metrics, it's has get it's way as an standard way of evaluating models in the industry.

AI is so good at optimizing metrics, so that's because we need to be very careful of choosing metrics.


### Pearson correlation coefficient
Pearon correlation is a very popular metric to measure how similar 2 variables are. As more similar a higher pearson correlation will have ```[-1, 1]```
Numpy has a method called ``` np.corrcoef(dataframe, rowvar=False) ``` in which we will obtain a matrix that will tell us the correlation between each one of the variables of the dataset.

A lot of statistical metrics relay of the square of the difference, so if we have outliers, the metric will be not be representative as in reality. In special pearson correlation is sensible to outliers.

A couple of samples which are outliers, would easily drop 0.2-0.5 points the pearson correlation between two variables.


### Notebook 
for more information see the notebook: https://www.kaggle.com/code/nicolsnnavarrete/getting-started-with-nlp-for-absolute-beginners/edit 




