import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment to debug warnings
# import warnings
# warnings.filterwarnings("error")


class FormulaFeatures:
    def __init__(self,
                 base_model=None,
                 metric=None,
                 max_iterations=None,
                 max_iterations_no_gain=None,
                 max_original_features=20,
                 target_type='regression',
                 test_square=False,
                 test_sqrt=False,
                 test_log=False,
                 verbose=0):
        """
        todo: allow users to specify the base model to use (DT by default), and the metric.
        We can determine if it's classification or regression from the y column (which they
        need to specify) and the metric -- check they agree.

        max_iterations_no_gain int: maximum number of iterations with no improvement in the metric of the top feature
        """

        self.base_model = base_model
        self.metric = metric
        self.max_iterations = np.inf if max_iterations is None else max_iterations
        self.max_iterations_no_gain = np.inf if max_iterations_no_gain is None else max_iterations_no_gain
        self.max_original_features = max_original_features
        self.target_type = target_type
        self.test_square = test_square
        self.test_sqrt = test_sqrt
        self.test_log = test_log
        self.verbose = verbose
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

        # All original features and the useful engineered features
        self.features_arr = None

        # Parallel to features_arr. Contains the formula to create the feature.
        self.feature_definitions = None

        # Parallel to features_arr. Contains the accuracy of the feature in a 1d model.
        self.feature_scores = None

        # Parallel to features_arr. Binary indicator if the feature is to be considered for future combinations
        self.use_feature = None

        # Parallel to features_arr. Each element contains a tuple with two elements: the values for the the feature in
        # the train data and the values for the feature in the test data.
        self.feature_values = None

        # The best score of any single original or engineered feature created to date.
        self.best_score_overall = -np.inf

    def fit(self, x, y):
        """
        This is a supervised feature engineering process, and as such requires the y column.
        """

        # Remove all Null and inf values to start
        x = self.__clean_data(x)

        # Create a train-test split within the passed data, which may itself be
        # a portion of the full data available
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.33, random_state=42)

        # For efficiency, limit the size of the training and testing data
        if len(self.x_train) > 50_000:
            self.x_train = self.x_train.sample(n=50_000)
            self.y_train = self.y_train.loc[self.x_train.index]
        if len(self.x_test) > 10_000:
            self.x_test = self.x_test.sample(n=10_000)
            self.y_test = self.y_test.loc[self.x_test.index]

        # Get the initial set of features, which are the features passed in.
        self.features_arr = x.columns.tolist()

        # Define the initial set of feature definitions. These are set as None, as they original features have no
        # definition to create them.
        self.feature_definitions = [None]*len(self.features_arr)

        # Store the values in each original & engineered column. At this point, we save the values of the original
        # columns. Each element is a tuple with the train values and the test values
        self.feature_values = []
        for col_idx, col_name in enumerate(x.columns):
            self.feature_values.append((self.x_train[col_name].values, self.x_test[col_name].values))

        # To start, we use all original features.
        self.use_feature = [True]*len(self.features_arr)

        # Examine the original columns
        self.__get_metrics_orig_features()
        self.__display_features_metrics(0)
        self.__get_best_feature_score()
        self.__reduce_original_features()

        # Get the initial engineered features based on pairs of original features.
        iteration_number = 1
        prev_num_features = len(self.features_arr)
        self.__combine_features(starting_idx=0)
        self.__display_features_metrics(iteration_number)
        self.__get_best_feature_score()

        # Loop, creating new combinations of features
        num_features = len(self.features_arr)
        iteration_number += 1
        while iteration_number <= self.max_iterations:
            self.__combine_features(starting_idx=prev_num_features)
            self.__display_features_metrics(iteration_number)
            self.__get_best_feature_score()

            new_num_features = len(self.features_arr)
            if new_num_features == num_features:
                break
            prev_num_features = num_features
            num_features = new_num_features
            iteration_number += 1

        self.__display_features_metrics("Final")

    def fit_transform(self, x):
        self.fit(x)
        x = x.copy()
        for feature_idx, feature_name in enumerate(self.features_arr):
            if feature_name not in x.columns:
                train_vals = pd.Series(self.feature_values[feature_idx][0], index=self.x_train.index)
                test_vals = pd.Series(self.feature_values[feature_idx][1], index=self.x_test.index)
                x[feature_name] = train_vals.append(test_vals)
        x = self.__clean_data(x)
        return x

    def transform(self, x):
        """
        Given a dataframe with the X features, return the same dataframe with the additional
        features determined in the fit process.
        """
        x = x.copy()
        for feature_idx, feature_name in enumerate(self.features_arr):
            if feature_name not in x.columns:
                feature_definition = self.feature_definitions[feature_idx]
                feat_1 = self.features_arr[feature_definition[0]]
                op = feature_definition[1]
                feat_2 = self.features_arr[feature_definition[2]]
                if op == 'add':
                    new_col = pd.DataFrame({feature_name: x[feat_1] + x[feat_2]})
                elif op == 'multiply':
                    new_col = pd.DataFrame({feature_name: x[feat_1] * x[feat_2]})
                elif op == 'subtract':
                    new_col = pd.DataFrame({feature_name: x[feat_1] - x[feat_2]})
                elif op == 'divide':
                    new_col = pd.DataFrame({feature_name: x[feat_1] / x[feat_2]})
                x = pd.concat([x, new_col], axis=1)

        x = self.__clean_data(x)
        return x

    def __get_metrics_orig_features(self):
        """
        Determine how strong each feature is individually in a 1d model.
        """

        self.feature_scores = []

        for col_name in self.features_arr:
            if self.target_type == 'regression':
                dt = DecisionTreeRegressor()  # todo: use the specified model type
                dt.fit(self.x_train[[col_name]], self.y_train)
                y_pred = dt.predict(self.x_test[[col_name]])
                r2 = r2_score(self.y_test, y_pred)
                self.feature_scores.append(r2)
            else:
                dt = DecisionTreeClassifier()  # todo: use the specified model type
                dt.fit(self.x_train[[col_name]], self.y_train)
                y_pred = dt.predict(self.x_test[[col_name]])
                f1 = f1_score(self.y_test, y_pred, average='macro')
                self.feature_scores.append(f1)

    def __display_features_metrics(self, iteration_number):
        if self.verbose >= 1:
            print()
            print("*********************************************************************************")
            print(f"After Iteration {iteration_number}. Features:")
            lst = list(zip(range(len(self.features_arr)), self.features_arr, self.feature_scores))
            best_score = -1
            for e in lst:
                if e[2] > best_score:
                    best_score = e[2]
                print(f"{e[0]:>4}: {round(e[2], 3):>8}, {e[1]}")
            print(f"Best Score: {best_score}")

    def display_features(self):
        lst = list(zip(range(len(self.features_arr)), self.features_arr, self.feature_scores, self.feature_definitions))
        for e in lst:
            print(f"{e[0]:>4}: {round(e[2], 3):>8}, {e[1]}")

    def __get_best_feature_score(self):
        best_score = -1
        for e in self.feature_scores:
            if e > best_score:
                best_score = e
        self.best_score_overall = best_score

    def __add_feature(self, best_score, best_operation, i, j, best_train_values, best_test_values):
        self.feature_scores.append(best_score)
        self.feature_values.append((best_train_values, best_test_values))
        self.feature_definitions.append((i, best_operation, j))
        self.use_feature.append(True)

    def __combine_features(self, starting_idx):
        """
        """

        new_elements = []
        best_score = -1
        best_operation = ""
        best_train_values = None
        best_test_values = None
        train_vals_i, train_vals_j, test_vals_i, test_vals_j = None, None, None, None

        def test_feat(operation):
            nonlocal best_score, best_operation, best_train_values, best_test_values
            nonlocal train_vals_i, train_vals_j, test_vals_i, test_vals_j

            temp_x_train = self.x_train.copy()
            temp_x_test = self.x_test.copy()

            if operation == 'add':
                temp_x_train['TEST'] = train_vals_i + train_vals_j
                temp_x_test['TEST']  = test_vals_i + test_vals_j
            elif operation == 'multiply':
                temp_x_train['TEST'] = train_vals_i * train_vals_j
                temp_x_test['TEST']  = test_vals_i  * test_vals_j
            elif operation == 'subtract':
                temp_x_train['TEST'] = train_vals_i - train_vals_j
                temp_x_test['TEST']  = test_vals_i  - test_vals_j
            elif operation == 'divide':
                temp_x_train['TEST'] = np.divide(train_vals_i, train_vals_j, out=np.zeros_like(train_vals_i), where=train_vals_j!=0)
                temp_x_test['TEST']  = np.divide(test_vals_i,  test_vals_j,  out=np.zeros_like(test_vals_i),  where=test_vals_j!=0)
            else:
                assert False

            temp_x_train = self.__clean_data(temp_x_train)
            temp_x_test  = self.__clean_data(temp_x_test)

            if self.target_type == 'regression':
                dt = DecisionTreeRegressor()
                dt.fit(temp_x_train[['TEST']], self.y_train)
                y_pred = dt.predict(temp_x_test[['TEST']])
                score = r2_score(self.y_test, y_pred)
            else:
                dt = DecisionTreeClassifier()
                dt.fit(temp_x_train[['TEST']], self.y_train)
                y_pred = dt.predict(temp_x_test[['TEST']])
                score = f1_score(self.y_test, y_pred, average='macro')

            score_parent_i = self.feature_scores[i]
            score_parent_j = self.feature_scores[j]
            if self.verbose >= 2:
                print(f"Columns: {col_i} and {col_j} -- {operation:<8} -- Score: {score}, parent score: {score_parent_i} and {score_parent_j}")
            if ((score > 0.1) or (score > self.best_score_overall)) and (score > score_parent_i) and (score > score_parent_j) and (score > best_score):
                best_score = score
                best_operation = operation
                best_train_values = temp_x_train['TEST'].values
                best_test_values = temp_x_test['TEST'].values

        unary_functions = ['none']
        if self.test_square:
            unary_functions.append('square')
        if self.test_sqrt:
            unary_functions.append('sqrt')
        if self.test_log:
            unary_functions.append('log')

        # Match all the new elements will all (old & new) elements. To do this, we match each new feature with all
        # features before it.
        for i in range(starting_idx, len(self.features_arr)):
            col_i = self.features_arr[i]
            for j in range(i):
                col_j = self.features_arr[j]
                if self.verbose >= 2:
                    print()
                    print("Testing columns: ", i, j)
                best_score = -1
                best_operation = ""
                best_train_values = []
                best_test_values = []

                for unary_i in unary_functions:
                    for unary_j in unary_functions:
                        train_vals_i = np.array(self.feature_values[i][0])
                        train_vals_j = np.array(self.feature_values[j][0])
                        test_vals_i  = np.array(self.feature_values[i][1])
                        test_vals_j  = np.array(self.feature_values[j][1])

                        test_feat("add")
                        test_feat('multiply')
                        test_feat("subtract")
                        test_feat('divide')
                if best_operation != '':
                    new_elements.append((self.features_arr[i], best_operation, self.features_arr[j]))
                    self.__add_feature(best_score, best_operation, i, j, best_train_values, best_test_values)

        start_current_iteration = len(self.features_arr)
        self.features_arr.extend(new_elements)
        self.__assess_new_features(new_elements, start_current_iteration)
        self.__check_arrays()

    def __assess_new_features(self, new_elements, starting_idx):
        new_cols_df = pd.DataFrame({x: self.feature_values[starting_idx + x_idx][0]
                                    for x, x_idx in zip(new_elements, range(len(new_elements)))})
        corr_matrix = new_cols_df.corr(method='spearman')
        arr_x, arr_y = np.where(np.triu(corr_matrix) > 0.95)

        remove_indexes_arr = []
        for e in list(zip(arr_x, arr_y)):
            if e[0] == e[1]:
                continue
            col_i = starting_idx + e[0]
            col_j = starting_idx + e[1]
            if self.feature_scores[col_i] > self.feature_scores[col_j]:
                remove_indexes_arr.append(col_j)
            else:
                remove_indexes_arr.append(col_i)

        self.__remove_features(remove_indexes_arr,
                               msg=f"Removing {len(remove_indexes_arr)} redundant features created during iteration")

    def __remove_features(self, remove_indexes_arr, msg):
        remove_indexes_arr = list(set(remove_indexes_arr))
        remove_indexes_arr.sort(reverse=True)
        if self.verbose >= 2:
            print()
            print(msg)
        for i in remove_indexes_arr:
            del(self.features_arr[i])
            del(self.feature_values[i])
            del(self.feature_definitions[i])
            del(self.feature_scores[i])

    def __check_arrays(self):
        assert len(self.features_arr) == \
               len(self.feature_values) == \
               len(self.feature_definitions) == \
               len(self.feature_scores)

    def __reduce_original_features(self):
        if len(self.features_arr) <= self.max_original_features:
            return
        bottom_features = np.argsort(self.feature_scores)[ : len(self.features_arr) - self.max_original_features]
        self.__remove_features(bottom_features,
                               msg=f"Excluding the least predictive {len(bottom_features)} features from examination")

    def __clean_data(self, df):
        df = df.fillna(0)
        df = df.replace([-np.inf, np.inf], 0)
        return df

    def plot_features(self):
        for feat_idx, feat_name in enumerate(self.features_arr):
            if self.target_type == 'regression':
                plt.scatter(x=self.feature_values[feat_idx][0], y=self.y_train)
            else:
                s = sns.boxplot(x=self.feature_values[feat_idx][0], y=self.y_train)
                #plt.boxplot(x=self.feature_values[feat_idx][0], y=self.y_train)
            s.xlabel = feat_name
            s.ylabel = 'Target'
            plt.title(f"Relationship of {feat_name} to Target")
            plt.show()


######################################################################################################################
# Methods to generate X data. These are similar to sklearn's make_classification() and make_regression(), but have
# a known f(x)


def generate_synthetic_x_data(num_rows=100_000, num_noise_cols=0, num_redundant_cols=0, seed=0):
    # The r2 on the full set of features and the individual features can vary greatly depending on the seed.
    np.random.seed(seed)
    random.seed(seed)

    a = np.random.random(num_rows)
    b = np.random.random(num_rows)
    d = np.random.random(num_rows)
    c = np.random.random(num_rows)

    # Ensure there are at least 4 columns
    data = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
    }

    for i in range(num_noise_cols):
        data[f"Noise_{i}"] = np.random.random(num_rows)

    for i in range(num_redundant_cols):
        noise = (np.random.random(num_rows) / 10.0)
        if i % 4 == 0:
            data[f"Redundant_A_{i}"] = data['a'] + noise
        if i % 4 == 1:
            data[f"Redundant_B_{i}"] = data['b'] + noise
        if i % 4 == 2:
            data[f"Redundant_C_{i}"] = data['c'] + noise
        if i % 4 == 3:
            data[f"Redundant_D_{i}"] = data['d'] + noise

    df = pd.DataFrame(data)

    return df


######################################################################################################################
# Methods to generate y column. These are similar to sklearn's make_classification() and make_regression(), but have
# a known f(x), relating the x columns to the y column

def generate_synthetic_y_formula_4_0(df):
    y = df['a'] + df['b']
    return y


def generate_synthetic_y_formula_4_1(df):
    y = ((((5.3 * df['a']) + df['b']) * df['c']) / df['d']) - ((5.4 * df['b']) - (2.1 * df['c']))
    return y


def generate_synthetic_y_formula_4_2(df):
    y = df['a'] * df['b'] * df['c'] + df['d']
    return y


def generate_synthetic_y_formula_4_3(df):
    y = df['a'] * df['b'] * df['c'] * df['d']
    return y


def generate_synthetic_y_formula_4_4(df):
    y = pd.Series(np.where(df['a'] > df['a'].median(),
                           df['a'] * df['b'],
                           df['c'] / df['d']))
    return y


def generate_synthetic_y_formula_4_5(df):
    # 'C' is used in both cases
    y = pd.Series(np.where(df['a'] > df['a'].median(),
                           df['a'] * df['b'] * df['c'],
                           df['c'] / df['d']))
    return y


def generate_synthetic_y_formula_4_6(df):
    # 'A^2' is used, and sqrt of 'C'
    y = pd.Series(np.where(df['a'] > df['a'].median(),
                           df['a'] * df['a'] * df['c'],
                           np.sqrt(df['c']) / df['d']))
    return y
