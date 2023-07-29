import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.datasets import fetch_openml

# Code from this project
from formula_features import FormulaFeatures, generate_synthetic_x_data
from formula_features import generate_synthetic_y_formula_4_0, \
                                generate_synthetic_y_formula_4_1, \
                                generate_synthetic_y_formula_4_2, \
                                generate_synthetic_y_formula_4_3, \
                                generate_synthetic_y_formula_4_4, \
                                generate_synthetic_y_formula_4_5, \
                                generate_synthetic_y_formula_4_6

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def clean_df(df):
    df = df.fillna(0)
    df = df.replace([-np.inf, np.inf], 0)
    return df


def test_r2(x_train, x_test, y_train, y_test):
    x_train = clean_df(x_train)
    x_test = clean_df(x_test)
    dt = DecisionTreeRegressor(max_leaf_nodes=10)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2


def test_f1(x_train, x_test, y_train, y_test):
    x_train = clean_df(x_train)
    x_test = clean_df(x_test)
    dt = DecisionTreeClassifier(max_leaf_nodes=10)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1


def demo_simple():
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


def demo_get_scores():
    data = fetch_openml('gas-drift')
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Drop all non-numeric columns. This is not necessary, but is done here for simplicity.
    x = x.select_dtypes(include=np.number)

    # Divide the data into train and test splits. For a more reliable measure of accuracy, cross validation may also
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


def demo_plot():
    data = fetch_openml('hill-valley')
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Drop all non-numeric columns. This is not necessary, but is done here for simplicity.
    x = x.select_dtypes(include=np.number)

    # Divide the data into train and test splits. For a more reliable measure of accuracy, cross validation may also
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
    ff.plot_features()


def test_synthetic():
    def test_dataset(num_noise_cols, num_redundant_cols, y_method, max_iterations):
        x = generate_synthetic_x_data(num_noise_cols=num_noise_cols, num_redundant_cols=num_redundant_cols)
        y = y_method(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        hashmarks = '#############################################################################################'
        print("\n\n")
        print(hashmarks)
        print(f"num_noise_cols: {num_noise_cols}")
        print(f"num_redundant_cols: {num_redundant_cols}")
        print(f"y_method: {y_method.__name__}")
        print(f"max_iterations: {max_iterations}")
        print(hashmarks)

        # Test using the original features
        r2 = test_r2(x_train, x_test, y_train, y_test)
        print(f"R2 score on original features: {r2}")

        ff = FormulaFeatures(max_iterations=max_iterations, verbose=1)
        ff.fit(x_train, y_train)
        x_train_extended = ff.transform(x_train)
        x_test_extended = ff.transform(x_test)
        display_df = x_test_extended.copy()
        display_df['Y'] = y_test.values

        # Test using the extended features
        r2 = test_r2(x_train_extended, x_test_extended, y_train, y_test)
        print(f"R2 score on extended features: {r2}")

    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_0, max_iterations=2)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_1, max_iterations=3)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_2, max_iterations=3)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_3, max_iterations=3)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_4, max_iterations=3)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_5, max_iterations=3)
    test_dataset(num_noise_cols=0, num_redundant_cols=0, y_method=generate_synthetic_y_formula_4_6, max_iterations=3)


def test_real():
    def test_dataset(dataset_name, max_iterations):
        data = fetch_openml(dataset_name)
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        target_type = 'regression'
        if y.dtype.name in ['category']:
            target_type = 'classification'

        print("\n\n")
        print('#############################################################################################')
        print(f"Dataset: {dataset_name}")
        print(f"Number of rows: {len(x)}")

        # Drop all non-numeric columns
        x = x.select_dtypes(include=np.number)
        if len(x.columns) < 2:
            print("At least two numeric columns required.")
            return

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Test using the original features
        if target_type == 'regression':
            orig_score = test_r2(x_train, x_test, y_train, y_test)
            print(f"R2 score on original features: {orig_score}")
        else:
            orig_score = test_f1(x_train, x_test, y_train, y_test)
            print(f"F1 (macro) score on original features: {orig_score}")

        ff = FormulaFeatures(
            max_iterations=max_iterations,
            max_original_features=10,
            target_type=target_type,
            verbose=0)
        ff.fit(x_train, y_train)
        x_train_extended = ff.transform(x_train)
        x_test_extended = ff.transform(x_test)
        display_df = x_test_extended.copy()
        display_df['Y'] = y_test.values

        # Test using the extended features
        if target_type == 'regression':
            extended_score = test_r2(x_train_extended, x_test_extended, y_train, y_test)
            print(f"R2 score on extended features: {extended_score}")
        else:
            extended_score = test_f1(x_train_extended, x_test_extended, y_train, y_test)
            print(f"F1 (macro) score on extended features: {extended_score}")
        return([dataset_name, target_type,  orig_score, extended_score])

    real_files = [
        'isolet',
        'bioresponse',
        'soybean',
        'micro-mass',
        'mfeat-karhunen',
        'Amazon_employee_access',
        'abalone',
        'cnae-9',
        'semeion',
        'vehicle',
        'satimage',
        'analcatdata_authorship',
        'breast-w',
        'SpeedDating',
        'eucalyptus',
        'vowel',
        'wall-robot-navigation',
        'credit-approval',
        'artificial-characters',
        'splice',
        'har',
        'cmc',
        'segment',
        'JapaneseVowels',
        'jm1',
        'gas-drift',
        'mushroom',
        'irish',
        'profb',
        'adult',
        'higgs',
        'anneal',
        'credit-g',
        'blood-transfusion-service-center',
        'monks-problems-2',
        'tic-tac-toe',
        'qsar-biodeg',
        'wdbc',
        'phoneme',
        'diabetes',
        'ozone-level-8hr',
        'hill-valley',
        'kc2',
        'eeg-eye-state',
        'climate-model-simulation-crashes',
        'spambase',
        'ilpd',
        'one-hundred-plants-margin',
        'banknote-authentication',
        'mozilla4',
        'electricity',
        'madelon',
        'scene',
        'musk',
        'nomao',
        'bank-marketing',
        'MagicTelescope',
        'Click_prediction_small',
        'PhishingWebsites',
        'nursery',
        'page-blocks',
        'hypothyroid',
        'yeast',
        'kropt',
        'CreditCardSubset',
        'shuttle',
        'Satellite',
        'baseball',
        'mc1',
        'pc1',
        'cardiotocography',
        'kr-vs-k',
        'volcanoes-a1',
        'wine-quality-white',
        'car-evaluation',
        'solar-flare',
        'allbp',
        'allrep',
        'dis',
        'car',
        'steel-plates-fault'
    ]

    results_arr = []
    for real_file in real_files:
        result = test_dataset(real_file, max_iterations=2)
        if result is not None:
            results_arr.append(result)
    results_df = pd.DataFrame(results_arr,
                              columns=['Dataset', 'Target Type', 'Score Original Features', 'Score Extended Features'])
    results_df['Improvement'] = results_df['Score Extended Features'] - results_df['Score Original Features']
    print(results_df)
    print(f"Number with improvement: {len(results_df[results_df['Improvement'] > 0.0])} out of {len(results_df)}")


if __name__ == '__main__':
    # Uncomment these as desired:
    # demo_simple()
    # demo_get_scores()
    demo_plot()
    # test_synthetic()
    #test_real()