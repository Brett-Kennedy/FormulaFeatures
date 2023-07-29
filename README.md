# FormulaFeatures
Feature engineering tool to efficiently create effective, arbitrarily complex arithmetic combinations of numeric features.

## Introduction

FormulaFeatures is named as such, as it allows generating features of sufficient complexity to be considered formulas. For example, given a dataset with columns A, B, C, D, E, F, the tool may generate features such as (A*B)/(D*E). However, it does so in a principled, step-wise manner, ensuring that each component of the final features created are justified. In most instances, the features engineered are combinations of two or three original features, though more where warranted, and not limited through hyperparameter selection. In this example, the tool would first determine both that A*B is a strong feature and (D*E) is as well before determining (A*B)/(D*E) is a stronger feature.

FormulaFeatures is a form of supervised feature engineering, considering the target column and producing a set of features specifically useful for predicting that target. This allows it to focus on a small number of engineered features, as simple or complex as necessary, without generating all possible combinations as with unsupervised methods. This supports both regression & classification targets. 

Even in the context of supervised feature engineering, there may be an explosion in the numbers of engineered features, resulting in long fit and tranform times, as well as producing more features than can be reasonably used by any downstream tasks, such as prediction, clustering, outlier detection. FormulaFeatuers is optimized to keep engineering time and the number of features returned tractable. 

## Unary Functions
By default, it does not incorporate unary functions, such as square, square root
unary functions. 
coeficients
conditions in f(x)

## Algorithm
The tool operates on the numeric features of a dataset. The first iteration, it examines each pair of original numeric features. For each, it considers four potential new features based on the four basic arithmetic operations (+, -, *, and /). If any perform better than both parent features, then the strongest of these is added to the set of features. Subsequent features consider combining all features generated in the previous iteration will all other features, again taking the strongest of these, if any. In this way, a practical number of new features are generated, all stronger than the previous features. Tests for correlation among the new set each iteration. May be correlated with one or both parent features, but will be an improvement. 

Each iteration, creates a more powerful set of features. This is a combination of combining weak features to make them stronger, and combining strong features to make these stronger as well. 

## Comparison to other Feature Engineering Tools
Formual is similar to [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), which will create arithmetic combinations of each pair of numeric features, based on simply operations (+, -, *, and /). This tool, however, differs in that:
- It will only generate features that are more predictive than either parent feature
- For any given pair of features, it will include only, if any, the combination based on the operator (+, -, *, or /) that results in the greatest predictive power
- It will continue looping for either a specified number of iterations, or so long as it is able to create more powerful features

The tool uses the fit-tranform pattern, also used by sklearn's PolynomialFeatures.

Internally uses a decision tree and either R2 or F1 (macro). Future will allow users to specify. 

Limits number of original columns considered when input has many. As it is common for datasets to contain hundreds of columns, creating even pairwise engineered features can invite overfitting as well as excessive time discovering the engineered features. 

Some interactions may be difficult to capture using arithmetic functions. Where there is a more complex relationship between pairs of features and the target column, it may be more appropriate to use [ikNN](https://github.com/Brett-Kennedy/ikNN).

## FormulaFeatures for Explainable AI (XAI)

the features are themselves a form of XAI. 

can use with interpretable model suchs as DT, NB, LR, GAM, and have greater accuracy. Can also be sparser (fewer features), though debatable if the toal int goes up or down. tune # iterations, # of engineered features used. 

Not likely to help with state of the art models for tabular data such as CatBoost, which tend to be able to capture feature interactions quite effectively in any case. In some cases, this may improve accuracy to a small degree, but it is intended for use primarily with interpretable models. 

## Example
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


  ## Example Notebook

give examples of true f(x), the features engineered. log, sqrt etc not fully necessary so long as monotonic. similar for coefficients. 

  ## Test Results

tests on synth & on real

does very well with synthetic. 

max_leaf_nodes = 10, so 10 rules. Manageable. 

will not be helpful if there are not interactions between the features. However, where there are interactions, discovering these can be quite informative. 

In some cases will generate no new features. In others will generate some, but this will not improve model accuracy. But, in some cases does, particularly with the shallow decision trees used here.


  
