> Random forests started a revolution in machine learning 20 years ago. For the first time, there was a fast and reliable algorithm which made almost no assumptions about the form of the data, and required almost no preprocessing. In today’s lesson, you’ll learn how a random forest really works, and how to build one from scratch. And, just as importantly, you’ll learn how to interpret random forests to better understand your data. 
> 	[Practical Deep Learning for Coders - 6: Random forests (fast.ai)](https://course.fast.ai/Lessons/lesson6.html)


##  **Random forest**
Random Forest is very popular, grewly in popularity from nearly 2000s. It's so elegant and is almost impossible to mess up. 
A lot of people says, "why you use ML? why you don't use something simplier like logistic regression?" And, in Industry, there are far more examples of people messing up using logistic regression than more advanced ML frameworks becuase, at a simple step that are doing wrong, the entire results don't matchup, nevertheless, random forest, is a simpler method that is nearly impossible to preprocess wrongly.

#### Example: Titanic dataset
For illustrate the power of random forest, first, Titanic dataset will be imported and the categorical features (such as sex, embarked, etc) will be converted as a category, that means, internally Pandas will treat the feature as numbers (which will mean a category, 0: femenine, 1: masculine).

```
from fastai.imports import *
np.set_printoptions(linewidth=130)

df = pd.read_csv(path/'train.csv')
tst_df = pd.read_csv(path/'test.csv')
modes = df.mode().iloc[0]

def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)

proc_data(df)
proc_data(tst_df)

cats=["Sex","Embarked"]
conts=['Age', 'SibSp', 'Parch', 'LogFare',"Pclass"]
dep="Survived"

df.Sex.head()
>> 0      male
1    female
2    female
3    female
4      male
Name: Sex, dtype: category
Categories (2, object): ['female', 'male']


df.Sex.cat.codes.head()
>> 0    1
1    0
2    0
3    0
4    1
dtype: int8

```

The key thing that points out is that with that, we're not going to need to create any categorical value, and we'll see why this is not necessary.

#### Binary split
A random forest is an ensamble of trees and a tree is an ensamble of binary splits.

A binary split is where all rows are placed into one of two groups, based on whether they're above or below some threshold of some column. 
For example, we could split the rows of our dataset into males and females, by using the threshold `0.5` and the column `Sex` (since the values in the column are `0` for `female` and `1` for `male`). We can use a plot to see how that would split up our data -- we'll use the [Seaborn](https://seaborn.pydata.org/) library, which is a layer on top of [matplotlib](https://matplotlib.org/) that makes some useful charts easier to create, and more aesthetically pleasing by default:

```
import seaborn as sns

fig,axs = plt.subplots(1,2, figsize=(11,5))
sns.barplot(data=df, y=dep, x="Sex", ax=axs[0]).set(title="Survival rate")
sns.countplot(data=df, x="Sex", ax=axs[1]).set(title="Histogram");
```
![[Pasted image 20230805113911.png]]
Binary split: Something that split the rows into two groups.

#### What happen if we use a model based on binary split by sex

Here we see that (on the left) if we split the data into males and females, we'd have groups that have very different survival rates: >70% for females, and <20% for males. We can also see (on the right) that the split would be reasonably even, with over 300 passengers (out of around 900) in each group.

We could create a very simple "model" which simply says that all females survive, and no males do. To do so, we better first split our data into a training and validation set, to see how accurate this approach turns out to be:

```
from numpy import random
from sklearn.model_selection import train_test_split

random.seed(42)
trn_df,val_df = train_test_split(df, test_size=0.25)
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)
```


(In the previous step we also replaced the categorical variables with their integer codes, since some of the models we'll be building in a moment require that.)

Now we can create our independent variables (the `x` variables) and dependent (the `y` variable):

```
def xs_y(df):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None

trn_xs,trn_y = xs_y(trn_df)
val_xs,val_y = xs_y(val_df)
```

  
Here's the predictions for our extremely simple model, where `female` is coded as `0`:
```
preds = val_xs.Sex==0
```

We'll use mean absolute error to measure how good this model is:
```
from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y, preds)

>> 0.21524663677130046
```

Could we do better?

Another example, what about fare? Fare is different respect to sex because fare is continous.
We could separate between 2 groups to see, for people to survive and people that did not survived. 
![[Pasted image 20230805115303.png]]

With the histogram we see that the majority of people had a log fare from 2. So we could create a separation based on that.
```
preds = val_xs.LogFare>2.7
mean_absolute_error(val_y, preds)

>> 0.336322869955157

```
Worst that our sex model.


#### A good binary split
The definition of a good binary split resides in the sparsity of the inside separated groups. If it's a good separation, almost all the variables at the one side will be pretty much the same, and the same way for the other side.

How similar are the things in the group? that's standard deviation *(std)*.
The std is useful for us to measure the quality of a binary split.
So the 'score' of a binary split will be the measure of the std times the number of samples for one side, plus the same calculation for the other side.
```
def _side_score(side, y):
    tot = side.sum()
    if tot<=1: return 0
    return y[side].std()*tot

    
def score(col, y, split):
    lhs = col<=split
    return (_side_score(lhs,y) + _side_score(~lhs,y))/len(y)


score(trn_xs["Sex"], trn_y, 0.5)
>> 0.40787530982063946

score(trn_xs["LogFare"], trn_y, 2.7)
>> 0.47180873952099694
```

With the previous code, we figured out that between LogFare and Sex, sex is better for splitting (since we're measuring *std*, the minor value is the best).

Would be nice if we built an automatic way of calculate by column the best separator. For that, we built a way of measure by separating the individual values that each column has.
```
def min_col(df, nm):
    col,y = df[nm],df[dep]
    unq = col.dropna().unique()
    scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
    idx = scores.argmin()
    return unq[idx],scores[idx]

cols = cats+conts
{o:min_col(trn_df, o) for o in cols}

>> {'Sex': (0, 0.40787530982063946),
 'Embarked': (0, 0.47883342573147836),
 'Age': (6.0, 0.478316717508991),
 'SibSp': (4, 0.4783740258817434),
 'Parch': (0, 0.4805296527841601),
 'LogFare': (2.4390808375825834, 0.4620823937736597),
 'Pclass': (2, 0.46048261885806596)}

```

ccording to this, `Sex<=0` is the best split we can use.

We've just re-invented the [OneR](https://link.springer.com/article/10.1023/A:1022631118932) classifier (or at least, a minor variant of it), which was found to be one of the most effective classifiers in real-world datasets, compared to the algorithms in use in 1993. Since it's so simple and surprisingly effective, it makes for a great _baseline_ -- that is, a starting point that you can use to compare your more sophisticated models to.

We found earlier that out OneR rule had an error of around `0.215`, so we'll keep that in mind as we try out more sophisticated approaches.


#### A step further - Decision tree
A 2R model is similar to 1R but with two binary splits, to do that, we can repeat the exact same piece of code but removing Sex from the coloumns to be evaluated and split the dataset into male and females.
Then we could run the code only in males and see the further separation.
```
cols.remove("Sex")
ismale = trn_df.Sex==1
males,females = trn_df[ismale],trn_df[~ismale]

>> {'Embarked': (0, 0.3875581870410906),
 'Age': (6.0, 0.3739828371010595),
 'SibSp': (4, 0.3875864227586273),
 'Parch': (0, 0.3874704821461959),
 'LogFare': (2.803360380906535, 0.3804856231758151),
 'Pclass': (1, 0.38155442004360934)}


```
For males, we see that the separation by Age is achieves the best MSE.
 
Then do the same thing for females.
```
{o:min_col(females, o) for o in cols}

>> {'Embarked': (0, 0.4295252982857327),
 'Age': (50.0, 0.4225927658431649),
 'SibSp': (4, 0.42319212059713535),
 'Parch': (3, 0.4193314500446158),
 'LogFare': (4.256321678298823, 0.41350598332911376),
 'Pclass': (2, 0.3335388911567601)}

```
And for females, we see that the separation by Pclass achieves the best MSE.

This is actually a decision tree, a series of binary separation spliting our data more and more, such as in the ends (or *leaf nodes* as we call it) we should have the strongest prediction as possible if survived.

Rather than writing that code manually, we can use `DecisionTreeClassifier`, from _sklearn_, which does exactly that for us:
```
from sklearn.tree import DecisionTreeClassifier, export_graphviz

m = DecisionTreeClassifier(max_leaf_nodes=4).fit(trn_xs, trn_y);
```
One handy feature or this class is that it provides a function for drawing a tree representing the rules:

```
import graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=2, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))



draw_tree(m, trn_xs, size=10)
```
![[Pasted image 20230805123207.png]]

We can see that it's found exactly the same splits as we did!

In this picture, the more orange nodes have a lower survival rate, and blue have higher survival. Each node shows how many rows ("_samples_") match that set of rules, and shows how many perish or survive ("_values_"). There's also something called "_gini_". That's another measure of impurity, and it's very similar to the `score()` we created earlier. It's defined as follows:
```
def gini(cond):
    act = df.loc[cond, dep]
    return 1 - act.mean()**2 - (1-act).mean()**2
```
What this calculates is the probability that, if you pick two rows from a group, you'll get the same `Survived` result each time. If the group is all the same, the probability is `1.0`, and `0.0` if they're all different:
```
gini(df.Sex=='female'), gini(df.Sex=='male')

mean_absolute_error(val_y, m.predict(val_xs))

>> 0.2242152466367713
```

It's a tiny bit worse. Since this is such a small dataset (we've only got around 200 rows in our validation set) this small difference isn't really meaningful. Perhaps we'll see better results if we create a bigger tree:
```
m = DecisionTreeClassifier(min_samples_leaf=50)
m.fit(trn_xs, trn_y)
draw_tree(m, trn_xs, size=12)
```
![[Pasted image 20230805123535.png]]

```
mean_absolute_error(val_y, m.predict(val_xs))

>> 0.18385650224215247
```

It looks like this is an improvement, although again it's a bit hard to tell with small datasets like this.

Decision trees don't care about the outliers, kinds of distributions, etc. It only separates the data. It's a great tool to create a baseline for tabular data. 


What if we wanted to make the model more accurate? Can we make the model deeper? I mean, it is possible but, overfit is dangerous, there are limitation in terms of how accurate a single decision tree could be. So, what can we do? We could do something that is amazing an fascinating,  a random forest *(by Leo Breiman)*.



### Random forest

We can't make the decision tree much bigger than the example above, since some leaf nodes already have only 50 rows in them. That's not a lot of data to make a prediction.

So how could we use bigger trees? One big insight came from Leo Breiman: what if we create lots of bigger trees, and take the average of their predictions? Taking the average prediction of a bunch of models in this way is known as [bagging](https://link.springer.com/article/10.1007/BF00058655).

The idea is that we want each model's predictions in the averaged ensemble to be uncorrelated with each other model. That way, if we average the predictions, the average will be equal to the true target value -- that's because the average of lots of uncorrelated random errors is zero. That's quite an amazing insight!

One way we can create a bunch of uncorrelated models is to train each of them on a different random subset of the data. Here's how we can create a tree on a random subset of the data:
```
def get_tree(prop=0.75):
    n = len(trn_y)
    idxs = random.choice(n, int(n*prop))
    return DecisionTreeClassifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs])
```

Now we can create as many trees as we want:
```
trees = [get_tree() for t in range(100)]
```

  
Our prediction will be the average of these trees' predictions:
```
all_probs = [t.predict(val_xs) for t in trees]
avg_probs = np.stack(all_probs).mean(0)

mean_absolute_error(val_y, avg_probs)

>> 0.2272645739910314
```
This is nearly identical to what `sklearn`'s `RandomForestClassifier` does. The main extra piece in a "real" random forest is that as well as choosing a random sample of data for each tree, it also picks a random subset of columns for each split. 

Here's how we repeat the above process with a random forest:
```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y);
mean_absolute_error(val_y, rf.predict(val_xs))

>> 0.18834080717488788
```


One particularly nice feature of random forests is they can tell us which independent variables were the most important in the model, using `feature_importances_`:
```
pd.DataFrame(dict(cols=trn_xs.columns, imp=m.feature_importances_)).plot('cols', 'imp', 'barh');
```
![[Pasted image 20230805124918.png]]

We can see that `Sex` is by far the most important predictor, with `Pclass` a distant second, and `LogFare` and `Age` behind that. In datasets with many columns, I generally recommend creating a feature importance plot as soon as possible, in order to find which columns are worth studying more closely. (Note also that we didn't really need to take the `log()` of `Fare`, since random forests only care about order, and `log()` doesn't change the order -- we only did it to make our graphs earlier easier to read.)

For details about deriving and understanding feature importances, and the many other important diagnostic tools provided by random forests, take a look at [chapter 8](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb) of [our book](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).


### Conclusion
So what can we take away from all this?

I think the first thing I'd note from this is that, clearly, more complex models aren't always better. Our "OneR" model, consisting of a single binary split, was nearly as good as our more complex models. Perhaps in practice a simple model like this might be much easier to use, and could be worth considering. Our random forest wasn't an improvement on the single decision tree at all.

So we should always be careful to benchmark simple models, as see if they're good enough for our needs. In practice, you will often find that simple models will have trouble providing adequate accuracy for more complex tasks, such as recommendation systems, NLP, computer vision, or multivariate time series. But there's no need to guess -- it's so easy to try a few different models, there's no reason not to give the simpler ones a go too!

Another thing I think we can take away is that random forests aren't actually that complicated at all. We were able to implement the key features of them in a notebook quite quickly. And they aren't sensitive to issues like normalization, interactions, or non-linear transformations, which make them extremely easy to work with, and hard to mess up!


