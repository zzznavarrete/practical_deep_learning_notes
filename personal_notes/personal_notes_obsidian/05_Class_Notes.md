
We're going to get deeper and deeper of how the neural networks works.
This lesson will be about building a tabular model, various tabular models from scratch.

### Notebook: 
The class started working in the notebook: https://www.kaggle.com/code/nicolsnnavarrete/linear-model-and-neural-net-from-scratch/.

#### data prep
First, the process of data cleaning must be done.
In pandas, there are very basics steps that we could do in order to get the dataset ready to next stage.
As for instance, the following command show the total number of NaNs variables in each column.
```
df.isna().sum
```

In case we don't want to delete the missing values, we could impute them using the mode of each column. The following method get the modes per column.
```
modes = df.mode().iloc[0]
```

Pandas has a very convinient method called 'fillna'. We could use that method to replace the null values with each mode per each column.
```
df.fillna(modes, inplace=True)
```
The previous is the most simpliest way to get ride of the missing values.

In most of the time, this is not the key differentiator most of the times in terms of model performance. The idea is not spend time in this aspects while creating a baseline model is ongoing.

<p>A way of justify impute instead of drop the data, is because the missing value most of the times will tell us more about that sample, that a non-existing sample because it was deleted</p>

Next step, always take a look of the numeric variables with describe the data and see their histogram plot per each one.
A thing that we could easily detect doing this kind of first inspection is finding distributions that are not "very easy to handle" for some models. For instance, there are models that do not perform well in exponentials distributions. So for get ride of that, we could apply a logaritmic distribution for those distributions. 
![[Pasted image 20230711225848.png]]


After, we could get a feeling inspecting the categorical variables.
```
df.describe(include=[object])
```

given that string features are not able to rawly beign mathematically evaluted. We use something called *dummy variables*, which basically is encoding categorical columns into new columns that marks with a number the equivalence of the string categorical.
```
df = pd.get_dummies(df, columns=['sex', 'pclass', 'embarked'])
```
![[Pasted image 20230711230109.png]]


ps: In this class we're not going to talk about feature engineering. 


#### Pytorch
There are many things afterwards that you could do in Numpy, but Pytorch does the same things and also get advantages of GPUs, so the professor think is better use Pytorch.

We could put the survivor column, the one that we wanted to predict, in a tensor:
```
t_dep = tensor(df.Survived)
```
we can now wrap the features values and conver them into a tensor
```
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
t_indeed = tensor(df[index_cols].values, dtype=torch.float)
t_indep
```
![[Pasted image 20230711230754.png]]

One of the most important attributes of a tensor is the shape (how many rows and how many columns it has)
```
t_indeep.shape
> torch.Size([891, 12])
```

The lenght of the shape it's called it's rank. The number of dimensions that it has.
vector = rank 1, matrix = rank 2, scalar = rank 0. (table is a rank 2 tensor)
```
len(t_indeep.shape)
> 2
```

we can now go ahead and multiply the features with some coefficients (model parameters).

### Setting up the model
The number of coefficient that we need is equal to the number of column we have. (features).

```
torch.manual_seed(442)
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5
coeffs
```

p.s: We place a manual seed to keep the experiments reproducible once we run the notebook again. Still in the professors opinion, it is not a suggestion always use a random each time we run an experiment, instead, it is good to understand how the data behaves in terms of it's relation with the models at each running, that could lead us to see that given that data the model varies a lot, or not. 

At using the technique of broadcasting, will allow us to perform very extensive calculations over GPU and in C-optimized code, so that's the reason of why we could use a "slow" programming language like Python to perform extensive computational tasks. 

https://numpy.org/doc/stable/user/basics.broadcasting.html
>  The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations.


So we perform the coefficients multiplications time the desired target
```
t_indep*coeffs
```
![[Pasted image 20230711232249.png]]

As we see, the first column of each vector of the tensor shows a notoriously larger number. That is because the first feature was 'age', so for optimization matters we're going to try to put all the features values between the same the same order of range.

For that purpose, we're going to take the max value of each feature and then 'normalize' each column by that.
```
vals, indices = t_indep.max(dim=0)
t_indep = t_indep / vals
```

So if we now look to the normalized variables with the coefficients, looks at the same range.
```
t_indep*coeffs
```
![[Pasted image 20230711232507.png]]

Obviously there are more techniques to normalize the data, usually it will have not much significant impact on the results, but still in case you're curious about the other technique, just google them.

Now that we already have the first model, we could make predictions.
```
preds = (t_indep*coeffs).sum(axis=1)
preds[:10]
```
![[Pasted image 20230711232726.png]]

For optimize the 'weights' or the values of each coefficients, we need to define a loss function for use as guidance at the gradient descent. 
A great loss function for regression problems is the mean absolute value.

```
loss = torch.abs(preds-t_dep).mean()
loss
```
![[Pasted image 20230711232921.png]]

As a way of doing it more structured, we define functions in order to reproduce the calculations over each change that we do in the coefficients weights.
```
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()

```

p.s: The professor encourages the don't change and don't delete cells of the notebook and also don't work outside of the notebook. This way, once I wanted to 'read' the notebook I'll remain my think process inside of the code flow that persisted there, other way, it's more cryptic to understand.


#### Doing a gradient descent step

First, we tell Pytorch that we wanted to change the gradient of the tensor.
In pytorch if there are an underscore at the end, means that the operation is inplace=True.
```
coeffs.requires_grad_()
```

Now we could calculate the loss, it's a gradient function.

