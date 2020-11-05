"""
Created on 2020-01-25

"""
import pandas as pd
import numpy as np
import time
from copy import copy as make_copy
from scipy.stats import mode
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from imblearn.over_sampling import SMOTE


class Preprocessor:
    def __init__(self):
        self.impute_etc = ExtraTreeClassifier()
        self.impute_dtc = DecisionTreeClassifier()
        self.impute_rfc = RandomForestClassifier()

    def get_data(self, url):
        print("Start getting data...")
        start = time.time()
        columns = [
            "age",
            "class of worker",
            "detailed industry recode",
            "detailed occupation recode",
            "education",
            "wage per hour",
            "enroll in edu inst last wk",
            "marital status",
            "major industry code",
            "major occupation code",
            "race",
            "hispanic origin",
            "sex",
            "member of a labor union",
            "reason for unemployment",
            "full or part time employment stat",
            "capital gains",
            "capital losses",
            "dividends from stocks",
            "tax filer stat",
            "region of previous residence",
            "state of previous residence",
            "detailed household and family stat",
            "detailed household summary in household",
            "instance weight",
            "migration code-change in msa",
            "migration code-change in reg",
            "migration code-move within reg",
            "live in this house 1 year ago",
            "migration prev res in sunbelt",
            "num persons worked for employer",
            "family members under 18",
            "country of birth father",
            "country of birth mother",
            "country of birth self",
            "citizenship",
            "own business or self employed",
            "fill inc questionnaire for veterans admin",
            "veterans benefits",
            "weeks worked in year",
            "year",
            "class",
        ]
        data = pd.read_csv(url, names=columns, na_values=" ?")
        # data = data.head(1000) # For testing purpose!!!
        for col in data.select_dtypes("O").columns:
            data[col] = data[col].astype("category")
        print(
            "Done getting data. Time taken = {:.1f}(s) \n".format(time.time() - start)
        )
        return data

    def get_null(self, data):
        contain_null = np.array(
            data.isnull().sum().to_frame()[data.isnull().sum().to_frame()[0] != 0].index
        )
        return contain_null

    def OnehotEncode(self, data, categorical_columns):
        df_1 = data.drop(columns=categorical_columns, axis=1)
        df_2 = pd.get_dummies(data[categorical_columns])
        df = pd.concat([df_1, df_2], axis=1, join="inner")
        return df

    def ImputeVoteClassifier(self, data, target_name):
        print("*" * 100 + "\n")
        print("Start imputing missing values for feature: {} \n".format(target_name))
        start = time.time()
        # Training set
        print("Generating training set...")
        train_data = data[data[target_name].notnull()].copy()
        train_target = train_data[target_name]
        train_data.drop(columns=[target_name], inplace=True)
        encoded_train = self.OnehotEncode(
            train_data, train_data.select_dtypes("category").columns
        )
        print("Done generating training set \n")
        # Testing set
        print("Generating testing set...")
        test_data = data[data[target_name].isnull()].copy()
        test_target = test_data[target_name]
        # Drop target var in testing set
        test_data.drop(columns=[target_name], inplace=True)
        encoded_test = self.OnehotEncode(
            test_data, test_data.select_dtypes("category").columns
        )
        print("Done generating testing set \n")
        # Fit data into base classifiers
        etc = make_copy(self.impute_etc)
        print("Fitting data into {}...".format(etc.__class__.__name__))
        etc.fit(encoded_train, train_target)
        etc_pred = etc.predict(encoded_test)

        dtc = make_copy(self.impute_dtc)
        print("Fitting data into {}...".format(dtc.__class__.__name__))
        dtc.fit(encoded_train, train_target)
        dtc_pred = dtc.predict(encoded_test)

        rfc = make_copy(self.impute_rfc)
        print("Fitting data into {}...".format(rfc.__class__.__name__))
        rfc.fit(encoded_train, train_target)
        rfc_pred = rfc.predict(encoded_test)

        # Finalize data
        print("Voting final predictions...")
        final_pred = np.array([])
        for i in range(0, len(test_target)):
            final_pred = np.append(
                final_pred, mode([etc_pred[i], dtc_pred[i], rfc_pred[i]])[0]
            )
        print(
            "Done voting and dumping final predictions into feature: {}. Time taken = {:.1f}(s) \n".format(
                target_name, time.time() - start
            )
        )
        print("\n" + "*" * 100)
        return final_pred

    def split_data(self, data, seed, re=False):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data["class"].values))
        data["class"] = lbl.fit_transform(list(data["class"].values))
        X, y = data.iloc[:, 0:-1], data.iloc[:, -1]
        X = self.OnehotEncode(X, X.select_dtypes("category").columns)
        X.columns = [col.replace("<", "_") for col in X.columns]
        # Train-Test split
        test_size = 0.3
        X_train_o, X_test, y_train_o, y_test = model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        # Resampling
        if re:
            resam = SMOTE(random_state=seed)
            resam.fit(X_train_o, y_train_o)
            X_train, y_train = resam.fit_resample(X_train_o, y_train_o)
            X_train = pd.DataFrame(X_train, columns=X_train_o.columns)
            y_train = pd.Series(y_train)
        else:
            X_train, y_train = X_train_o, y_train_o
        return X_train, y_train, X_test, y_test
