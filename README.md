# FormulaFeatures
A feature engineering tool to efficiently create effective, arbitrarily-complex arithmetic combinations of numeric features. 

For a description, see the article on Medium: [FormulaFeatures: A Tool to Generate Highly Predictive Features for Interpretable Models](https://medium.com/towards-data-science/formulafeatures-a-tool-to-generate-highly-predictive-features-for-interpretable-models-e18aab45e96d)

## Introduction

FormulaFeatures is intended for use primarily with interpretable models, such as shallow decision trees, where having single highly concise and highly predictive features can aid greatly with the interpretability and accuracy of the models. 

FormulaFeatures is a tool to engineer features based on the original features, and is often helpful for increasing the accuracy of predictors on tabular data. 

FormulaFeatures is named as such as it allows generating features of sufficient complexity to be considered simple formulas. For example, given a dataset with columns A, B, C, D, E, F, the tool may generate features such as (A * B) / (D * E). However, it does so in a principled, step-wise manner, ensuring that each component of the final features created is justified. In most instances, the features engineered are combinations of just two or three original features, though may be based on more where warranted (and not limited through hyperparameter selection). In this example, the tool would first determine both that A * B is a strong feature and that D * E is as well before determining if (A * B) / (D * E) is stronger still and including it if so. 

FormulaFeatures is more likely to engineer features such as A * B than (A * B) / (D * E), as simpler features will be favoured, keeping the engineered features just as complex as is warranted. 

FormulaFeatures is a form of supervised feature engineering, considering the target column and producing a set of features specifically useful for predicting that target. This allows it to focus on a small number of engineered features, as simple or complex as necessary, without generating all possible combinations, as is done with unsupervised methods. This supports both regression & classification targets. 

Even within the context of supervised feature engineering, (depending on the specific approach used) there may be an explosion in the numbers of engineered features, resulting in long fit and transform times, as well as producing more features than can be reasonably used by any downstream tasks, such as prediction, clustering, or outlier detection. FormulaFeatures is optimized to keep both engineering time and the number of features returned tractable, and its algorithm is designed to limit the numbers of features generated. 

## Unary Functions
By default, FormulaFeatures does not incorporate unary functions, such as square, square root, or log. In some cases, it may be that a feature such as A<sup>2</sup> / B predicts the target better than the equivalent form without the square operator: A / B. However, including unary operators can lead to misleading features if not substantially correct and may not significantly increase the accuracy of any models using them. When using, for example, decision trees (or tree-based models, such as Random Forest, CatBoost, XGBoost, LGBM, Bayesian Additive Regression Trees, etc), so long as there is a monotonic relationship between the features and target, which most unary functions maintain (with exceptions such as sin and cos, which may reasonably be used where cyclical patterns are strongly suspected), there will not be any change in the final accuracy of the model. In terms of explanatory power, simpler functions can often capture nearly as much as more complex functions and are more comprehensible to people examining them. 

A similar argument may be made for including coefficients in engineered features. A feature such as 5.3A + 1.4B may capture the relationship between A and B with Y better than the simpler A+B, but the coefficients are often unnecessary, prone to be calculated incorrectly, and inscrutable even where approximately correct. In the case of multiplication and division operations, the coefficients are most likely irrelevant, as 5.3A * 1.4B will be functionally equivalent to A*B for most purposes, as the difference is a constant which can be divided out. Again, there is a monotonic relationship with and without the coefficients, and thus the features are equivalant to models such as decision trees that are concerned only with the ordering of feature values, not their specific values. This may effect distance-based models such as SVN and kNN, but will not tend to affect tree-based models.

## Algorithm
The tool operates on the numeric features of a dataset. In the first iteration, it examines each pair of original numeric features. For each, it considers four potential new features based on the four basic arithmetic operations (+, -, *, and /). We limit the features produced to these operations for greater interpretability and to ensure the process is tractable. If any perform better than both parent features, then the strongest of these is added to the set of features. 

Subsequent iterations then consider combining all features generated in the previous iteration will all other features. That is, we consider all the new possible pairs of features, without re-examining the pairs of features that existed in previous iterations. At each step we, again take the strongest of these new features, if any are stronger than both parents. In this way, a practical number of new features are generated, all stronger than the previous features. 

At the end of each iteration, the correlation among the features created this iteration is examined, and where two or more features that are highly correlated are found, only the strongest is kept, removing the others (there can be sets of three or more correlated features). This process does allow for new features that are correlated with features created in previous iterations, as the new features will be stronger, while the earlier features will be less complex, and quite potentially still useful in later iterations. 

In this way, each iteration creates a more powerful set of features than the previous. This is a combination of combining weak features to make them stronger (and more likely useful in downstream tasks), as well as combining strong features to make these stronger, creating what are most likely the most predictive features. 

The process uses 1D models (models utilizing a single feature) to evaluate each feature. This has a number of advantages, in particular:

- 1D models are quite fast both to train and test, which allows the fitting process to execute much quicker than many other approaches
- 1D models are simple and so may reasonably operate on smaller samples of the data, further improving efficiency
- Testing with strictly single features at a time (original and engineered features) eliminates searching for effective combinations of features, which is compuationally expensive.
- This ensures all features useful in themselves, so supports the features being a form of XAI in themselves (described further below). 

One effect of using 1D models is, almost all engineered features will have global significance, which is often desirable, but it does mean the tool can miss generating additional features that would be useful only in specific sub-spaces. This is, at least potentially with some datasets, a limitation.

Setting the tool to execute with verbose=1 or verbose=2 allows viewing the process in greater detail. A simple example is sketched here as well.

Assume we start with a dataset with features A, B, and C and that this is a regression problem. Future versions will support more metrics, but the currently-available version internally uses R2 for regression problems and F1 (macro) for classification problems. (Using these metrics will still identify the most predictive features, even if optimized for slightly different metrics than the final model will use.) So, in this case we begin with calculating the R2 for each original feature, training a decision tree using only feature A, then only B, then only C. This may give the following R2 scores:
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
Here there are two operations that have a higher R2 score than either parent feature (A or B): + and *. We take the highest of these, which is A + B, and add this to the set of features. We do the same for A & B and B & C. In most cases, no feature will be added, but often one is. After the first iteration we may have:

```
A       0.43
B       0.02
C      -1.23
A + B   0.54
B / C   0.32
```
We then, in the next iteration, take the two features just added, and try combining them with all other features, including each other. After this we may have:
```
A                   0.43
B                   0.02
C                  -1.23
A + B               0.54
B / C               0.32
(A + B) - C         0.56
(A + B) * (B / C)   0.66
```
This continues until there is no longer improvement, or a limit specified by a hyperparameter, called max_iterations, is reached. 

The tool does limit the number of original columns considered when the input data has many columns. As it is common for datasets to contain hundreds of columns, creating even pairwise engineered features can invite overfitting as well as excessive time creating and evaluating the potential engineered features. So, where datasets have large numbers of columns, only the most predictive are considered after the first iteration. The subsequent iterations perform as normal; there is simply a reduction of the original features used at the beginning. 

Note: the tool provides strictly feature engineering, and may return more features than are necessary or useful for some models. As such, subsequent feature selection will still be necessary in most cases. 

## Comparison to other Feature Engineering Tools
 
The tool uses the fit-tranform pattern, the same as that used by sklearn's PolynomialFeatures and many other feature engineering tools. And so, it is easy to substitute this tool for others to determine which is the most useful for any given project. 

FormulaFeatures is similar to [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), which is unsupervised and will create arithmetic combinations of each pair of original numeric features, based on simply operations (+, -, *, and /). This tool, however, differs in that:

- It will generate far fewer features, but each that it generates will be useful
- It will only generate features that are more predictive than either parent feature
- For any given pair of features, it will include only, if any, the combination based on the operator (+, -, *, or /) that results in the greatest predictive power
- It will continue looping for either a specified number of iterations, or so long as it is able to create more powerful features, and so can create more powerful features than ArithmeticFeatures, which is limited to features based on pairs of original features. 

Another popular feature engineering tool based on arithmetic operations is AutoFeat, which works similarly to [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), also in an unsupervised manner, so will create many more features. This increases the need for feature selection, but as feature selection is generally done in any case, AutoFeat may also generate useful features for any given project. That is, [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), autoFeat, and other unsupervised methods amy work quite well for any given project, and simply require more feature selection to be performed. 

The current project focuses more on XAI goals and memory efficiency. 

Some interactions may be difficult to capture using arithmetic functions. Where there is a more complex relationship between pairs of features and the target column, it may be more appropriate to use [ikNN](https://github.com/Brett-Kennedy/ikNN).

## FormulaFeatures for Explainable AI (XAI)

One interesting property of FormulaFeatures is that the features generated are themselves a form of XAI. Examining the features engineered, particularly those with strong scores, can provide insights into the data, and how the features interact with the target. Though, this is an approximation of the true f(x), it can be a useful approximation. For example, with original features A and B, it may be known that A and B are both positively associated with the target, but if A + B is significantly more strongly associated, this may be informative that there is true additive relationship between the features.

The goal of the tool is to create a small set of powerful features, which allow potentially interpretable models, such as Decision Trees, Decision Tables, Rule Sets, Rule Lists, Naive Bayes, ikNN, Genetic Decision Trees, Additive Decsision Trees, and Generalized Additive Models to produce accurate models with few features, allowing them to be quite interpretable. Though in some cases it may be debatable if the overall interpretability is increased or decreased when using more than a few complex features. In these cases, it may be advised to tune the maximum number of iterations (and hense maximum complexity of the engineered features) or the number of engineered features used in an interpretable model. Or to simply manually filter any features produced by FormulaFeatures if they appear too complex for a given audience. 

With Decision Trees, the engineered features generated tend to be put at the top of the trees (as these are the most powerful features, best able to maximize information gain), but no single feature can split the data perfectly at any step. Other feaetures are used lower in the tree, which tend to be simpler engineered features (based only only two original features), or the original features. On the whole, this can produce fairly interpretable decision trees.

Where there are feature interactions, decsion trees can often deal with these well, but by creating very deep trees, where the data space is divided into each combination of the two (or more features). This can work well in terms of accuracy (though it does mean that splits lower in the tree are based on fewer and fewer samples, so become progressively less reliable). But, large decision trees are effectively incomprehensible. They are very difficult for people to assess. Much of the power of FormulaFeatures is in generating single features that can allow single (or a small number of) splits in a decision tree, that can remove the need for many nodes if using only the original features. 

FormulaFeatures is not likely to help with state of the art models for tabular data such as CatBoost, which tend to be able to capture feature interactions quite effectively in any case. In some cases, this may improve accuracy to a small degree, but it is intended for use primarily with interpretable models. 

## Scaling the features

For some model types, such as SVM and kNN models, the features engineered by FormulaFeatures may be on quite different scales than the orginal features, and may need to be scaled. 

## Instalation

The tool uses a single .py file, which may be simply downloaded and used. It has no dependencies other than numpy, pandas, matplotlib, and seaborn. 

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

Getting the feature scores may be useful for understanding the features generated and for feature selection. 

```python
  data = fetch_openml('gas-drift')
  x = pd.DataFrame(data.data, columns=data.feature_names)
  y = data.target
  
  # Drop all non-numeric columns. This is not necessary, but is done here for simplicity.
  x = x.select_dtypes(include=np.number)
  
  # Divide the data into train and test splits. For a more reliable measure of accuracy, cross validation may 
  # be used. This is done here for simplicity.
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
  
  ff = FormulaFeatures(
      max_iterations=2,
      max_original_features=10,
      target_type='classification',
      verbose=1)
  ff.fit(x_train, y_train)
  x_train_extended = ff.transform(x_train)
  x_test_extended = ff.transform(x_test)
  
  display_df = x_test_extended.copy()
  display_df['Y'] = y_test.values
  print(display_df.head())
  
  # Test using the extended features
  extended_score = test_f1(x_train_extended, x_test_extended, y_train, y_test)
  print(f"F1 (macro) score on extended features: {extended_score}")
  
  # Get a summary of the features engineered and their scores based on 1D models
  ff.display_features()
```

This will produce the following report:

This lists each feature index, F1 macro score, and feature name.

```
   0:    0.438, V9
   1:    0.417, V65
   2:    0.412, V67
   3:    0.412, V68
   4:    0.412, V69
   5:    0.404, V70
   6:    0.409, V73
   7:    0.409, V75
   8:    0.409, V76
   9:    0.414, V78
  10:    0.447, ('V65', 'divide', 'V9')
  11:    0.465, ('V67', 'divide', 'V9')
  12:    0.422, ('V67', 'subtract', 'V65')
  13:    0.424, ('V68', 'multiply', 'V65')
  14:    0.489, ('V70', 'divide', 'V9')
  15:    0.477, ('V73', 'subtract', 'V65')
  16:    0.456, ('V75', 'divide', 'V9')
  17:     0.45, ('V75', 'divide', 'V67')
  18:    0.487, ('V78', 'divide', 'V9')
  19:    0.422, ('V78', 'divide', 'V65')
  20:    0.512, (('V67', 'divide', 'V9'), 'multiply', ('V65', 'divide', 'V9'))
  21:    0.449, (('V67', 'subtract', 'V65'), 'divide', 'V9')
  22:     0.45, (('V68', 'multiply', 'V65'), 'subtract', 'V9')
  23:    0.435, (('V68', 'multiply', 'V65'), 'multiply', ('V67', 'subtract', 'V65'))
  24:    0.535, (('V73', 'subtract', 'V65'), 'multiply', 'V9')
  25:    0.545, (('V73', 'subtract', 'V65'), 'multiply', 'V78')
  26:    0.466, (('V75', 'divide', 'V9'), 'subtract', ('V67', 'divide', 'V9'))
  27:    0.525, (('V75', 'divide', 'V67'), 'divide', ('V73', 'subtract', 'V65'))
  28:    0.519, (('V78', 'divide', 'V9'), 'multiply', ('V65', 'divide', 'V9'))
  29:    0.518, (('V78', 'divide', 'V9'), 'divide', ('V75', 'divide', 'V67'))
  30:    0.495, (('V78', 'divide', 'V65'), 'subtract', ('V70', 'divide', 'V9'))
  31:    0.463, (('V78', 'divide', 'V65'), 'add', ('V75', 'divide', 'V9'))
```
This includes the original features for context. 

## Example Plotting the Engineered Features

Plotting the features is also supported and can also be useful for understanding the relationships of the features to the target. An example is provided in the demo python file included in this repository. It is the same as the above example, but includes the line:

```python
  ff.plot_features()
```

In the case of regression targets, the tool presents a scatter plot mapping each feature to the target. In the case of classification targets, the tool presents a boxplot, giving the distribution of a feature broken down by class label. It is often the case that the orginal features show little difference in distributions per class, while engineered features can show a distinct difference. For example, one feature generated, (V99 / V64) - (V99 / V42) shows a strong separation:

![Example](https://github.com/Brett-Kennedy/FormulaFeatures/blob/main/images/eng_feat_2.png). 

This is typical of the features engineered; while each has an imperfect seperation, each is strong, often much more so than for the original features. 

## Example File

In the demo folder, a python file called demo.py has been provided, which includes examples working with synthetic and real data. This includes the tests with real datasets from OpenML described below. 

## Test Results

Testing was performed on synthetic and real data. The tool performed very well on the synthetic data, but this provides more debugging and testing than meaningful evaluation. For real data, a set of 80 random datasets from OpenML were selected, though only those having at least two numeric features could be included, leaving 69 files. Testing consisted of performing a simple, single train-test split on the data, then traing and evaluating a model on the numeric feature both before and after engineering additional features. For classification datasets Macro F1 was used, and for regression, R2. As the tool is designed for use with interpretable models, a decision tree (either scikit-learn's DecisionTreeClassifer or DecisionTreeRegressor) was used, setting max_leaf_nodes = 10 (corresponding to 10 induced rules) to ensure an interpretable model.

In many cases, the tool provided for no improvement or only slight improvements in the accuracy of the shallow decision trees, as is expected. No feature engineering technique will work in all cases. More informative is that the tool led to significant increases inaccuracy an impressive number of times. This is without tuning or feature selection, which can further improve the utility of the tool. As well, using interpretable models other than shallow decision trees will give different results. 

The tool will not be helpful if there are not interactions between the features. However, where there are interactions, discovering these can be quite informative in itself. But, often there are no detectable interactions, and the tool will generate no new features. In other cases, it will generate some, but this will not improve model accuracy. But, in some cases it does noteably improve the model accuracy with the shallow decision trees used here.

We can very often get better results limiting max_iterations to 2, compared to 3. This is a hyperparameter, and must be tuned like any other. But, for most datasets, using 2 or 3 works well, while with others, setting much higher, or to None (which allows the process to continue so long as it can produce more effective features), may work well. 

For all files, the time engineer the new features was under two minutes, even with many of the test files have hundreds of columns and many thousands of rows. 

```
                             Dataset     Target Type  Score Original Features  Score Extended Features  Improvement
                             isolet  classification                 0.248937                 0.256383     0.007446
                        bioresponse  classification                 0.750731                 0.752049     0.001318
                         micro-mass  classification                 0.750323                 0.775414     0.025091
                     mfeat-karhunen  classification                 0.665896                 0.765000     0.099104
                            abalone  classification                 0.127990                 0.122068    -0.005922
                             cnae-9  classification                 0.718347                 0.746005     0.027658
                            semeion  classification                 0.517225                 0.554099     0.036874
                            vehicle  classification                 0.674043                 0.726682     0.052639
                           satimage  classification                 0.754425                 0.699809    -0.054616
             analcatdata_authorship  classification                 0.906717                 0.896386    -0.010331
                          breast-w   classification                 0.946253                 0.939917    -0.006336
                       SpeedDating   classification                 0.601201                 0.608292     0.007091
                        eucalyptus   classification                 0.525395                 0.560346     0.034951
                             vowel   classification                 0.431700                 0.461311     0.029612
             wall-robot-navigation   classification                 0.975749                 0.975749     0.000000
                   credit-approval   classification                 0.748106                 0.710384    -0.037722
             artificial-characters   classification                 0.289557                 0.322401     0.032843
                               har   classification                 0.870952                 0.870943    -0.000009
                               cmc   classification                 0.492402                 0.402663    -0.089739
                           segment   classification                 0.917215                 0.934663     0.017447
                    JapaneseVowels   classification                 0.573279                 0.686150     0.112871
                               jm1   classification                 0.534338                 0.544699     0.010362
                         gas-drift   classification                 0.741395                 0.833291     0.091896
                             irish   classification                 0.659593                 0.610964    -0.048630
                             profb   classification                 0.558397                 0.544389    -0.014008
                             adult   classification                 0.588593                 0.588593     0.000000
                             higgs       regression                 0.122135                 0.122135     0.000000
                            anneal   classification                 0.609106                 0.619520     0.010414
                          credit-g   classification                 0.528565                 0.488953    -0.039612
  blood-transfusion-service-center   classification                 0.639358                 0.621569    -0.017789
                       qsar-biodeg   classification                 0.778677                 0.804669     0.025991
                              wdbc   classification                 0.936013                 0.947647     0.011634
                           phoneme   classification                 0.756816                 0.743363    -0.013452
                          diabetes   classification                 0.716462                 0.661243    -0.055219
                   ozone-level-8hr   classification                 0.575845                 0.591788     0.015943
                       hill-valley   classification                 0.527126                 0.743144     0.216018
                               kc2   classification                 0.683200                 0.683200     0.000000
                     eeg-eye-state   classification                 0.664768                 0.713241     0.048474
  climate-model-simulation-crashes   classification                 0.470414                 0.643538     0.173124
                          spambase   classification                 0.891185                 0.912952     0.021766
                              ilpd   classification                 0.566124                 0.607570     0.041446
         one-hundred-plants-margin   classification                 0.058236                 0.055540    -0.002697
           banknote-authentication   classification                 0.952441                 0.995498     0.043057
                          mozilla4   classification                 0.925336                 0.924435    -0.000901
                       electricity   classification                 0.778510                 0.787295     0.008785
                           madelon   classification                 0.712804                 0.760869     0.048066
                             scene   classification                 0.669597                 0.710699     0.041102
                              musk   classification                 0.810198                 0.842862     0.032664
                             nomao   classification                 0.905249                 0.911504     0.006255
                    bank-marketing   classification                 0.658861                 0.645403    -0.013458
                    MagicTelescope   classification                 0.780897                 0.807032     0.026136
            Click_prediction_small   classification                 0.494579                 0.494416    -0.000163
                       page-blocks   classification                 0.669840                 0.816824     0.146985
                       hypothyroid   classification                 0.924111                 0.907954    -0.016157
                             yeast   classification                 0.445225                 0.487207     0.041982
                  CreditCardSubset   classification                 0.785395                 0.803868     0.018473
                           shuttle   classification                 0.651774                 0.514898    -0.136875
                         Satellite   classification                 0.886097                 0.902899     0.016803
                          baseball   classification                 0.627791                 0.701683     0.073892
                               mc1   classification                 0.705530                 0.665058    -0.040471
                               pc1   classification                 0.473381                 0.550434     0.077052
                  cardiotocography   classification                 1.000000                 0.991587    -0.008413
                           kr-vs-k   classification                 0.097765                 0.116503     0.018738
                      volcanoes-a1   classification                 0.366350                 0.327768    -0.038582
                wine-quality-white   classification                 0.252580                 0.251460    -0.001120
                             allbp   classification                 0.555988                 0.553157    -0.002831
                            allrep   classification                 0.279349                 0.288094     0.008745
                               dis   classification                 0.696957                 0.563886    -0.133071
                steel-plates-fault   classification                 1.000000                 1.000000     0.000000
```

The model performed better with than without FormulaFeatures feature engineering 49 out of 69 cases. Note though, only one dataset in the random sample was a regression problem, so this case is not well-tested here. 

Some noteworthy examples are: 
- Japanese Vowels improved from .57 to .68
- gas-drift improved from .74 to .83
- hill-valley improved from .52 to .74
- climate-model-simulation-crashes improved from .47 to .64
- banknote-authentication improved from .95 to .99
- page-blocks improved from .66 to .81
