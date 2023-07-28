# FormulaFeatures
Feature engineering tool to efficiently create effective, arbitrarily complex arithmetic combinations of numeric features.

supervised feature engineering
support regression & classification targets
optimized to keep engineering time and the number of features returned tractable. 
unary functions. 
coeficients
conditions in f(x)

The tool operates on the numeric features of a dataset. The first iteration, it examines each pair of original numeric features. For each, it considers four potential new features based on the four basic arithmetic operations (+, -, *, and /). If any perform better than both parent features, then the strongest of these is added to the set of features. Subsequent features consider combining all features generated in the previous iteration will all other features, again taking the strongest of these, if any. In this way, a practical number of new features are generated, all stronger than the previous features.

This is similar to [ArithmeticFeatures](https://github.com/Brett-Kennedy/ArithmeticFeatures), which will create arithmetic combinations of each pair of numeric features, based on simply operations (+, -, *, and /). This tool, however, differs in that:
- It will only generate features that are more predictive than either parent feature
- For any given pair of features, it will include only, if any, the combination based on the operator (+, -, *, or /) that results in the greatest predictive power
- It will continue looping for either a specified number of iterations, or so long as it is able to create more powerful features

The tool uses the fit-tranform pattern, also used by sklearn's PolynomialFeatures

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

  ## Test Results

  
