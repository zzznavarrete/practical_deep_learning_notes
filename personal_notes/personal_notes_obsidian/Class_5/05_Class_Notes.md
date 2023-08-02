
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

Pandas has a very convenient method called 'fillna'. We could use that method to replace the null values with each mode per each column.
```
df.fillna(modes, inplace=True)
```
The previous is the most simplest way to get ride of the missing values.

In most of the time, this is not the key differentiator most of the times in terms of model performance. The idea is not spend time in this aspects while creating a baseline model is ongoing.

A way of justify impute instead of drop the data, is because the missing value most of the times will tell us more about that sample, that a non-existing sample because it was deleted.

Next step, always take a look of the numeric variables with describe the data and see their histogram plot per each one.
A thing that we could easily detect doing this kind of first inspection is finding distributions that are not "very easy to handle" for some models. For instance, there are models that do not perform well in exponentials distributions. So for get ride of that, we could apply a logaritmic distribution for those distributions. 
![[Pasted image 20230711225848.png]]


After, we could get a feeling inspecting the categorical variables.
```
df.describe(include=[object])
```

given that string features are not able to rawly being mathematically evaluated. We use something called *dummy variables*, which basically is encoding categorical columns into new columns that marks with a number the equivalence of the string categorical.
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
we can now wrap the features values and convert them into a tensor
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

![[Pasted image 20230801215355.png]]

Then, we assign the 'calc_loss' function to the variable 'loss' and pass the coefficients as parameters. Python will remember that these coefficients need to be adjusted based on the defined loss function, which, in this case, aims to minimize the mean square error. The goal is to make each coefficient respond to movements in a way that reduces the overall error.
```
loss = calc_loss(coeffs, t_indep, t_dep)
loss

>> tensor(0.5382, grad_fn=<MeanBackward0>)
```

After see what do we got as output of the 'backward' function of the inner variable 'loss', apparently, seems that, in order to minimize the loss, we need to increase the value of the first coefficient *(see the difference between the value previous and after performing the loss gradient evaluation)*:

```
loss.backward()
coeffs.grad
```
![[Pasted image 20230801215337.png]]

In order to do that, we could arbitrarily define a value to move the coefficients in this direction.
```
with torch.no_grad():
	coeffs.sub_(coeffs.grad * 0.1)
	print(calc_loss(coeffs, t_indep, t_dep))

>> tensor(0.5197)
```
And as we can see, now our loss has improve with respect it's previous value.

Now we have everything that we need to train a linear model.

### Train the linear model
```
from fastai.data.transforms import RandomSplitter
trn_split, val_split = RandomSplitter(seed=42)(df)

trn_indep, val_indep = t_indep[trn_split], t_indep[val_split]
trn_dep, val_dep = t_dep[trn_split], t_dep[val_split]

len(trn_indep), len(val_indep)

>> (713, 178)
```

defining functions
```
def update_coeffs(coeffs, lr): coeffs.sub_(coeffs.grad * lr)

def one_epoch(coeffs, lr):
	loss = calc_loss(coeffs, trn_indep, trn_dep)
	loss.backward()
	with torch.no_grad(): update_coeffs(coeffs, lr)
	print(f'{loss:.3f}', end=";")

def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()


def train_model(epochs=30, lr=0.01):
	torch.manual_seed(442)
	coeffs = init_coeffs()
	for i in range(epochs): one_epoch(coeffs, lr=lr)
	return coeffs
```

Now we train the model
```
coeffs = train_model(epochs=18, lr=0.02)

>> 0.536, 0.528, 0.521, 0.513, 0.506, 0.498, 0.491, 0.484, 0.476, 0.468, 0.460, 0.453, 0.445,0.400, 0.348, 0.301, 0.294.
```


Finally, we could see what the coefficients are attached to each variable:
```
def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))

show_coeffs()
```
![[Pasted image 20230801221610.png]]

### Measuring accuracy

Accuracy is a critical matter in terms of model building, for that so we could create the following method.
We'll assume that any passenger with a score of over `0.5` is predicted to survive. So that means we're correct for each row where `preds>0.5` is the same as the dependent variable
```
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)

>> tensor(0.7865)
```
And with that, essentially we have constructed a model that could predict with an accuracy of 78% the titanic survival.


### Using sigmoid
If we see the predictions, there are times in which we'll predict over 1 and below 0, that do not make sense since we want to have 0 or 1 as limits. 
Here's a great function, the sigmoid function. 
![[Pasted image 20230801224008.png]]
The sigmoid function maps values to a range between 0 and 1, often used for probabilities in machine learning. Its formula is f(x) = 1 / (1 + e^(-x)), forming an S-shaped curve.

Now we could calculate the predictions filtered through this sigmoid function.
```
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))

coeffs = train_model(lr=2)

>> 0.510; 0.327; 0.294; 0.207; 0.201; 0.199; 0.198; 0.197; 0.196; 0.196; 0.196; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194;
```
The loss has improved a lot!, let's check the accuracy
```
acc(coeffs)
>> tensor(0.8258)
```
That's improved too! Here's the coefficients of our trained model:
![[Pasted image 20230801225305.png]]

### Using matrix product.

We can make things quite a bit neater...

Take a look at the inner-most calculation we're doing to get the predictions:

```
(val_indep*coeffs).sum(axis=1)
```
Multiplying elements together and then adding across rows is identical to doing a matrix-vector product! Python uses the `@` operator to indicate matrix products, and is supported by PyTorch tensors. Therefore, we can replicate the above calculate more simply like so:
```
val_indep@coeffs
```
It also turns out that this is much faster, because matrix products in PyTorch are very highly optimised.

Let's use this to replace how `calc_preds` works:
```
def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)
```
In order to do matrix-matrix products (which we'll need in the next section), we need to turn `coeffs` into a column vector (i.e. a matrix with a single column), which we can do by passing a second argument `1` to `torch.rand()`, indicating that we want our coefficients to have one column:
```
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()
```
We'll also need to turn our dependent variable into a column vector, which we can do by indexing the column dimension with the special value `None`, which tells PyTorch to add a new dimension in this position:
```
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]
```
We can now train our model as before and confirm we get identical outputs...:
```
coeffs = train_model(lr=100)
>> 0.512; 0.323; 0.290; 0.205; 0.200; 0.198; 0.197; 0.197; 0.196; 0.196; 0.196; 0.195; 0.195; 0.195; 0.195; 0.195; 0.195; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194; 0.194;
```
...and identical accuracy:
```
acc(coeffs)
>> tensor(0.8258)
```


Now that we have it implemented through matrix multiplication, we're ready to use neural networks.


### Neural network
We've now got what we need to implement our neural network.

First, we'll need to create coefficients for each of our layers. Our first set of coefficients will take our `n_coeff` inputs, and create `n_hidden` outputs. We can choose whatever `n_hidden` we like -- a higher number gives our network more flexibility, but makes it slower and harder to train. So we need a matrix of size `n_coeff` by `n_hidden`. We'll divide these coefficients by `n_hidden` so that when we sum them up in the next layer we'll end up with similar magnitude numbers to what we started with.

Then our second layer will need to take the `n_hidden` inputs and create a single output, so that means we need a `n_hidden` by `1` matrix there. The second layer will also need a constant term added.

```
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()
```

Now we have our coefficients, we can create our neural net. The key steps are the two matrix products, `indeps@l1` and `res@l2` (where `res` is the output of the first layer). The first layer output is passed to `F.relu` (that's our non-linearity), and the second is passed to `torch.sigmoid` as before.
```
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
```
Finally, now that we have more than one set of coefficients, we need to add a loop to update each one:
```
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```
That's it -- we're now ready to train our model!
```
coeffs = train_model(lr=1.4)

>> 0.543; 0.400; 0.260; 0.390; 0.221; 0.211; 0.197; 0.195; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.193; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192; 0.192;
```

```
acc(coeffs)
>> tensor(0.8258)
```
In this case our neural net isn't showing better results than the linear model. That's not surprising; this dataset is very small and very simple, and isn't the kind of thing we'd expect to see neural networks excel at. Furthermore, our validation set is too small to reliably see much accuracy difference. But the key thing is that we now know exactly what a real neural net looks like!


### Deep learning
The neural net in the previous section only uses one hidden layer, so it doesn't count as "deep" learning. But we can use the exact same technique to make our neural net deep, by adding more matrix multiplications.

First, we'll need to create additional coefficients for each layer:
```
def init_coeffs():
    hiddens = [10, 10]  # <-- set this to the size of each hidden layer you want
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts
```
You'll notice here that there's a lot of messy constants to get the random numbers in just the right ranges. When you train the model in a moment, you'll see that the tiniest changes to these initialisations can cause our model to fail to train at all! This is a key reason that deep learning failed to make much progress in the early days -- it's very finicky to get a good starting point for our coefficients. Nowadays, we have ways to deal with that, which we'll learn about in other notebooks.

Our deep learning `calc_preds` looks much the same as before, but now we loop through each layer, instead of listing them separately:
```
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)
```

We also need a minor update to `update_coeffs` since we've got `layers` and `consts` separated now:
```
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```
Let's train our model...
```
coeffs = train_model(lr=4)
>> 0.521; 0.483; 0.427; 0.379; 0.379; 0.379; 0.379; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.378; 0.377; 0.376; 0.371; 0.333; 0.239; 0.224; 0.208; 0.204; 0.203; 0.203; 0.207; 0.197; 0.196; 0.195;
```
...and check its accuracy:
```
acc(coeffs)
>> tensor(0.8258)
```


### Why you should use a framework

Independently everything could be done from scratch. Nowadays there are no reasons of why not take advantages of the frameworks that are available (ie, fastai) and ensure that every initial and basic step are programmed correctly.
- Best practices are handled for you automatically -- fast.ai has done thousands of hours of experiments to figure out what the best settings are for you
- Less time getting set up, which means more time to try out your new ideas
- Each idea you try will be less work, because fastai and PyTorch will do the many of the menial bits for you
- You can always drop down from fastai to PyTorch if you need to customise any part (or drop down from the fastai Application API to the fastai mid or low tier APIs), or even drop down from PyTorch to plain python for deep customisation.

Let's see how that looks in practice. We'll start by doing the same library setup as in the "from scratch" notebook:

```
from pathlib import Path
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/titanic')
    !pip install -Uqq fastai
else:
    import zipfile,kaggle
    path = Path('titanic')
    if not path.exists():
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)
```

```
from fastai.tabular.all import *

pd.options.display.float_format = '{:.2f}'.format
set_seed(42)
```


#### prep the data

```
df = pd.read_csv(path/'train.csv')
```
When you do everything from scratch, every bit of feature engineering requires a whole lot of work, since you have to think about things like dummy variables, normalization, missing values, and so on. But with fastai that's all done for you. So let's go wild and create lots of new features! We'll use a bunch of the most interesting ones from this fantastic [Titanic feature engineering notebook](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial/) (and be sure to click that link and upvote that notebook if you like it to thank the author for their hard work!)
```
def add_features(df):
    df['LogFare'] = np.log1p(df['Fare'])
    df['Deck'] = df.Cabin.str[0].map(dict(A="ABC", B="ABC", C="ABC", D="DE", E="DE", F="FG", G="FG"))
    df['Family'] = df.SibSp+df.Parch
    df['Alone'] = df.Family==0
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Title'] = df.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Title'] = df.Title.map(dict(Mr="Mr",Miss="Miss",Mrs="Mrs",Master="Master"))

add_features(df)
```

As we discussed in the last notebook, we can use `RandomSplitter` to separate out the training and validation sets:
```
splits = RandomSplitter(seed=42)(df)
```

Now the entire process of getting the data ready for training requires just this one cell!:

```
dls = TabularPandas(
    df, splits=splits,
    procs = [Categorify, FillMissing, Normalize],
    cat_names=["Sex","Pclass","Embarked","Deck", "Title"],
    cont_names=['Age', 'SibSp', 'Parch', 'LogFare', 'Alone', 'TicketFreq', 'Family'],
    y_names="Survived", y_block = CategoryBlock(),
).dataloaders(path=".")
```

Here's what each of the parameters means:

- Use `splits` for indices of training and validation sets: `splits=splits,`
- Turn strings into categories, fill missing values in numeric columns with the median, normalize all numeric columns: `procs = [Categorify, FillMissing, Normalize],`
- These are the categorical independent variables: `cat_names=["Sex","Pclass","Embarked","Deck", "Title"],`
- These are the continuous independent variables: `cont_names=['Age', 'SibSp', 'Parch', 'LogFare', 'Alone', 'TicketFreq', 'Family'],`
- This is the dependent variable: `y_names="Survived",`
- The dependent variable is categorical (so build a classification model, not a regression model): `y_block = CategoryBlock(),`

```
learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
```
You'll notice we didn't have to do any messing around to try to find a set of random coefficients that will train correctly -- that's all handled automatically.

One handy feature that fastai can also tell us what learning rate to use:

```
learn.lr_find(suggest_funcs=(slide, valley))
```
The two colored points are both reasonable choices for a learning rate. I'll pick somewhere between the two (0.03) and train for a few epochs:
![[Pasted image 20230801232314.png]]

```
learn.fit(16, lr=0.03)
```
We've got a similar accuracy to our previous "from scratch" model -- which isn't too surprising, since as we discussed, this dataset is too small and simple to really see much difference. A simple linear model already does a pretty good job. But that's OK -- the goal here is to show you how to get started with deep learning and understand how it really works, and the best way to do that is on small and easy to understand datasets.


### Ensambling

Since it's so easy to create a model now, it's easier to play with more advanced modeling approaches. For instance, we can create five separate models, each trained from different random starting points, and average them. This is the simplest approach of [ensembling](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/) models, which combines multiple models to generate predictions that are better than any of the single models in the ensemble.

To create our ensemble, first we copy the three steps we used above to create and train a model, and apply it to the test set:
```
def ensemble():
    learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
    with learn.no_bar(),learn.no_logging(): learn.fit(16, lr=0.03)
    return learn.get_preds(dl=tst_dl)[0]
```

Now we run this five times, and collect the results into a list:
```
learns = [ensemble() for _ in range(5)]
```

We stack this predictions together and take their average predictions:
```
ens_preds = torch.stack(learns).mean(0)
```

### Final thoughts
As you can see, using fastai and PyTorch made things much easier than doing it from scratch, but it also hid away a lot of the details. So if you only ever use a framework, you're not going to as fully understand what's going on under the hood. That understanding can be really helpful when it comes to debugging and improving your models. But do use fastai when you're creating models on Kaggle or in "real life", because otherwise you're not taking advantage of all the research that's gone into optimising the models for you, and you'll end up spending more time debugging and implementing menial boiler-plate than actually solving the real problem!

