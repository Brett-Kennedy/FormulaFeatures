# FormulaFeatures
Feature engineering tool to efficiently create effective, arbitrarily-complex arithmetic combinations of numeric features.

## Introduction

FormulaFeatures is named as such, as it allows generating features of sufficient complexity to be considered formulas. For example, given a dataset with columns A, B, C, D, E, F, the tool may generate features such as (A*B)/(D*E). However, it does so in a principled, step-wise manner, ensuring that each component of the final features created are justified. In most instances, the features engineered are combinations of two or three original features, though more where warranted and not limited through hyperparameter selection. In this example, the tool would first determine both that A*B is a strong feature and that (D*E) is as well before determining if (A*B)/(D*E) is stronger still and including it if so.

FormulaFeatures is a form of supervised feature engineering, considering the target column and producing a set of features specifically useful for predicting that target. This allows it to focus on a small number of engineered features, as simple or complex as necessary, without generating all possible combinations as with unsupervised methods. This supports both regression & classification targets. 

Even within the context of supervised feature engineering, there may be an explosion in the numbers of engineered features, resulting in long fit and transform times, as well as producing more features than can be reasonably used by any downstream tasks, such as prediction, clustering, or outlier detection. FormulaFeatures is optimized to keep both engineering time and the number of features returned tractable. 

## Unary Functions
By default, FormulaFeatures does not incorporate unary functions, such as square, square root, or log. In some cases, it may be that a feature such as A<sup>2</sup> / B predicts the target better than the equivalent without the square operator: A / B. However, including unary operators can lead to misleading features if not substantially correct and may not significantly increase the accuracy of any models using them. When using, for example, decision trees, so long as there is a monotonic relationship between the features and target, which most unary functions maintain, there will not be any change in the final accuracy of the model. In terms of explanatory power, simpler functions can often capture nearly as much as more complex functions and are more comprehensible to any people examining them. 

A similar argument may be made for including coefficients in engineered features. A feature such as 5.3A + 1.4B may capture the relationship between A and B with Y better than the simpler A+B, but the coefficients are often unnecessary, prone to be calculated incorrectly, and inscrutable even where approximately correct. In the case of multiplication and division operations, the coefficients are most likely irrelevant, as 5.3A * 1.4B will be equivalent to A*B for most purposes, as the difference is a constant which can be divided out. 

## Algorithm
The tool operates on the numeric features of a dataset. In the first iteration, it examines each pair of original numeric features. For each, it considers four potential new features based on the four basic arithmetic operations (+, -, *, and /). If any perform better than both parent features, then the strongest of these is added to the set of features. Subsequent iterations then consider combining all features generated in the previous iteration will all other features, again taking the strongest of these, if any. In this way, a practical number of new features are generated, all stronger than the previous features. 

At the end of each iteration, the correlation among the features created this iteration is examined, and where two or more features that are highly correlated where created, only the strongest is kept, removing the others. This process does allow for new features that are correlated with features created in previous iterations, as the new features will be stronger, while the earlier features will be less complex, and quite potentially still useful in later iterations. 

In this way, each iteration creates a more powerful set of features than the previous. This is a combination of combining weak features to make them stronger, and more likely useful in a downstream task, as well as combining strong features to make these stronger. 

The process uses 1D models (models utilizing a single feature) to evaluate each feature. This has a number of advantages, in particular:

- 1D models are quite fast both to train and test, which allows the fitting process to execute much quicker
- 1D models are simpler may reasonably operate on smaller samples of the data, further improving efficiency
- Using single features eliminates searching for effective combinations of features
- This ensures all features useful in themselves, so supports the features being XAI in themselves. 

One effect of using 1D models is, almost all engineered features have global significance, which is often desirable, but it does mean the tool can miss generating features useful only in specific sub-spaces.

Setting the tool to execute with verbose=1 or verbose=2 allows viewing the process in greater detail. A simple example is sketche here as well.

Assume we start with a dataset with features A, B, and C and that this is a regression problem. Future versions will support more metrics, but the currently-available version internally uses R2 for regression problems and F1 (macro) for classification problems. So, in this case we begin with calculating the R2 for each original feature, training a decision tree using only feature A, then only B, then only C. This may give the following R2 scores:
```
A   0.43
B   0.02
C  -1.23
```
We then consider the combinations of these, which are: A & B, A & C, and B & C. For each we try the four arithmetic operations: +, *, -, and /. When examining A & B, assume we get the following R2 scores:
```
A + B  0.54
A * B  0.44
A - B  0.21
A / B  -0.01
```
Here there are two operations that have a higher R2 score than either parent feature (A or B), + and *. We take the highest of these, which is A + B, and add this to the set of features. We do the same for A & B and B & C. In most cases, no feature is added, but often one is. After the first iteration we may have:

```
A       0.43
B       0.02
C      -1.23
A + B   0.54
B / C   0.32
```
We then take the two features just added, and try combining them with all other features, including each other. After this we may have:
```
A                   0.43
B                   0.02
C                  -1.23
A + B               0.54
B / C               0.32
(A + B) - C         0.56
(A + B) * (B / C)   0.66
```
This continues until there is no longer improvement, or a limit specified by a hyperparameter, commonly max_iterations, is reached. 

The tool does limits number of original columns considered when input has many. As it is common for datasets to contain hundreds of columns, creating even pairwise engineered features can invite overfitting as well as excessive time discovering the engineered features. So, where datasets have large numbers of columns, only the most predictive are considered after the first iteration. 

Note: the tool provides strictly feature engineering, and may return more features than are necessary or useful for some models. As such, subsequent feature selection will still be necessary in most cases. 


## Comparison to other Feature Engineering Tools

The tool uses the fit-tranform pattern the same as that used by sklearn's PolynomialFeatures and many other feature engineering tools. And so, it is easy to substitute this tool for others to determine which is the most useful for any given project. 

FormulaFeatures is similar to [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), which will create arithmetic combinations of each pair of original numeric features, based on simply operations (+, -, *, and /). This tool, however, differs in that:

- It will only generate features that are more predictive than either parent feature
- For any given pair of features, it will include only, if any, the combination based on the operator (+, -, *, or /) that results in the greatest predictive power
- It will continue looping for either a specified number of iterations, or so long as it is able to create more powerful features, and so can create more powerful features than ArithmeticFeatures, which is limited to features based on pairs of original features. 

Some interactions may be difficult to capture using arithmetic functions. Where there is a more complex relationship between pairs of features and the target column, it may be more appropriate to use [ikNN](https://github.com/Brett-Kennedy/ikNN).

Another popular feature engineering tool based on arithmetic operations is AutoFeat, which works similarly, but in an unsupervised manner, so will create many more features. This increases the need for feature selection, as this is generally done in any case, AutoFeat may also generate useful features for any given project. 

## FormulaFeatures for Explainable AI (XAI)

One interesting property of FormulaFeaturs is that the features generated are themselves a form of XAI. You can see the relationships between the features and target more clearly. Though, this is an approximation of the true f(x), it can be a useful approximation. 

including the unary functions and coefficients may increase interpretability in some cases, but can also be misleading, often wrong. And models can be almost as accurate without. For example, decision trees require only a monotonic relationship between the features and the true f(x). The relatively simpler features generated by default, generally enhance interpretability. However, enabling the unary features can produce better results in some cases and hyperparameters settings allow this. As well, for other unary operations, this may be done before calling fit().

can use with interpretable model suchs as DT, NB, LR, GAM, and have greater accuracy. Can also be sparser (fewer features), though debatable if the toal int goes up or down. tune # iterations, # of engineered features used. For some model types, such as SVM, kNN, etc, the features engineered here may be on quite different scales than the orginal features, and may need to be scaled. 

Not likely to help with state of the art models for tabular data such as CatBoost, which tend to be able to capture feature interactions quite effectively in any case. In some cases, this may improve accuracy to a small degree, but it is intended for use primarily with interpretable models. 

with DTs, will tend to put the engineered features at the top (the most powerful, but no single features can split the data perfectly at any step), original features lower in the tree. 

It's arguable if DT or other potentially interpretable model is still interpretable is uses more than a few engineered features of more than a certain level of complexity. What's interpretable is very much context-specific. Can limit # iterations, or # eng (and original) features used. 

## Simple Example
```python
  import pandas as pd
  from sklearn.datasets import load_iris
  from formula_features import FormulaFeatures

  iris = load_iris()
  x, y = iris.data, iris.target
  x = pd.DataFrame(x, columns=iris.feature_names)

  x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

  ff = FormulaFeatures()
  ff.fit(x_train, y_train)
  x_train_extended = ff.transform(x_train)
  x_test_extended = ff.transform(x_test)

  dt = DecisionTreeClassifier(max_depth=4, random_state=0)
  dt.fit(x_train_extended, y_train)
  y_pred = dt.predict(x_test_extended)
```

## Example Getting Feature Scores

## Example Plotting the Engineered Features


## Example Notebook

give examples of true f(x), the features engineered. log, sqrt etc not fully necessary so long as monotonic. similar for coefficients. 

Uses synth & openml datasets

## Test Results

tests on synth & on real

does very well with synthetic. 

max_leaf_nodes = 10, so 10 rules. Manageable. 

will not be helpful if there are not interactions between the features. However, where there are interactions, discovering these can be quite informative. 

In some cases will generate no new features. In others will generate some, but this will not improve model accuracy. But, in some cases does, particularly with the shallow decision trees used here.

Can get better results limiting max_iterations to 2 compared to 3. This is a hyperparameter, and must be tuned like any other. But, for most datasets, using 2 or 3 works well, while with others, setting much higher, or to None (which allows the process to continue so long as it can produce more effective features), may work well. 

In many cases, the tools provided for no improvement or only slight improvements in the accuracy of the shallow decision trees, as is expected. No feature engineering technique will work in all cases. More informative is that the tool led to significant increases inaccuracy an impressive number of times. This is without tuning or feature selection, which can further improve the utility of the tool. As well, using interpretably models other than shallow decision trees will give different results. 

For all files, the time engineer the new features was under two minutes, even with many of the test files have hundreds of columns and many thousands of rows. 

Japanese Vowels .57 to .68
gas-drift .74 to .83
hill-valley .52 to .74
climate-model-simulation-crashes .47 to .64
