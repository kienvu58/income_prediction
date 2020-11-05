"""
Created on 2020-01-26

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt


class Mymetrics:
    def __init__(self):
        pass

    def accuracy(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_test, y_pred)
        # print('Accuracy: {:.2f}%'.format(acc * 100))
        return acc

    def roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
        return fpr, tpr, threshold

    def auc_score(self, y_test, y_pred_proba):
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)
        # print('Classifier AUC: {:.2f}%'.format(roc_auc*100))
        return roc_auc

    def precision_score(self, y_test, y_pred):
        precision_scr = metrics.precision_score(y_test, y_pred)
        # print('Precision score is {:.2f}'.format(float(precision_scr)))
        return precision_scr

    def recall_score(self, y_test, y_pred):
        recall_scr = metrics.recall_score(y_test, y_pred)
        # print('Recall score is {:.2f}'.format(float(recall_scr)))
        return recall_scr


class Myvisualization(Mymetrics):
    def __init__(self):
        pass

    def roc_auc_viz(self, y_test, y_pred_proba):
        fpr, tpr, threshold = self.roc_curve(y_test, y_pred_proba)
        roc_auc = self.auc_score(y_test, y_pred_proba)
        gini_score = 2 * roc_auc - 1
        plt.title("Receiver Operating Characteristic")
        plt.plot(
            fpr,
            tpr,
            "b",
            label="AUC = {:.2f} and GINI = {:.2f}".format(roc_auc, gini_score),
        )
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        pass


class Model(Mymetrics):
    def __init__(self):
        self.clf_0 = DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=10, min_samples_split=10
        )
        self.clf_1 = AdaBoostClassifier(
            n_estimators=256,
            learning_rate=0.05,
        )
        self.clf_2 = RandomForestClassifier(
            n_estimators=256,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="auto",
            max_depth=2,
            class_weight="balanced",
        )
        self.clf_3 = LogisticRegression(
            penalty="l2", solver="liblinear", C=1000, class_weight="balanced"
        )

    def generate_oof(self, clf, X_trainset, y_trainset, X_testset, n_fold, seed):
        print("Start getting out of fold set for {}...".format(clf.__class__.__name__))
        folds = model_selection.StratifiedKFold(
            n_splits=n_fold, random_state=seed, shuffle=True
        )
        start = time.time()
        oof_train = np.zeros((X_trainset.shape[0],))
        oof_test = np.zeros((X_testset.shape[0],))
        oof_test_skf = np.empty((n_fold, X_testset.shape[0]))
        for i, (train_idx, test_idx) in enumerate(folds.split(X_trainset, y_trainset)):
            X_train, X_test = X_trainset.iloc[train_idx], X_trainset.iloc[test_idx]
            y_train, y_test = y_trainset.iloc[train_idx], y_trainset.iloc[test_idx]
            clf.fit(X_train, y_train)
            oof_train[test_idx] = clf.predict_proba(X_test)[:, 1]
            print(
                "Base classifier {}: AUC = {:.2f} | Accuracy = {:.2f} | Precision score: {:.2f} | Recall score: {:.2f}".format(
                    clf.__class__.__name__ + "_" + str(i),
                    self.auc_score(y_test, clf.predict_proba(X_test)[:, 1]),
                    self.accuracy(y_test, clf.predict(X_test)),
                    float(self.precision_score(y_test, clf.predict(X_test))),
                    float(self.recall_score(y_test, clf.predict(X_test))),
                )
            )
            oof_test_skf[i, :] = clf.predict_proba(X_testset)[:, 1]
            # joblib.dump(clf, clf.__class__.__name__+'_'+str(i)+'_'+'.pkl')
        oof_test[:] = oof_test_skf.mean(axis=0)
        print(
            "Done getting out of fold set for {}. Time taken = {:.1f}(s) \n".format(
                clf.__class__.__name__, time.time() - start
            )
        )
        oof_train = oof_train.ravel()
        oof_test = oof_test.ravel()
        return oof_train, oof_test

    # def generate_metadata(self, oof_df ):
    #     metadata = pd.concat(oof_df, axis=1)
    #     return metadata

    def generate_metadata(
        self, X_train, y_train, X_test, y_test, clf_list, generate_oof, n_fold, seed
    ):
        oof_train = {}
        oof_test = {}
        for clf in clf_list:
            clf_oof_train, clf_oof_test = generate_oof(
                clf, X_train, y_train, X_test, n_fold, seed
            )
            oof_train[clf.__class__.__name__] = clf_oof_train
            oof_test[clf.__class__.__name__] = clf_oof_test
        meta_train = pd.DataFrame(oof_train)
        meta_test = pd.DataFrame(oof_test)
        return meta_train, meta_test

    def model_predict(self, model, X_train, y_train, X_test, y_test, seed):
        if "random_state" in model.get_params().keys():
            model.set_params(random_state=seed)
        print("Start fitting Meta classifier...")
        start = time.time()
        model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Accuracy
        acc = self.accuracy(y_test, y_pred)
        roc_auc = self.auc_score(y_test, y_pred_proba)
        precision_scr = self.precision_score(y_test, y_pred)
        recall_scr = self.recall_score(y_test, y_pred)
        print("Accuracy: {:.2f}%".format(acc * 100))
        print("Meta Classifier AUC: {:.2f}%".format(roc_auc * 100))
        print("Precision score: {:.2f}".format(float(precision_scr)))
        print("Recall score: {:.2f}".format(float(recall_scr)))
        print(
            "Done fitting meta classifier. Time taken = {:.1f}(s) \n".format(
                time.time() - start
            )
        )
        return model

    def cross_validate(self, model, X, y, seed):
        kfold = model_selection.StratifiedKFold(
            n_splits=4, shuffle=True, random_state=seed
        )
        print("Start cross validating Meta classifier...")
        results = model_selection.cross_val_score(
            model, X, y, scoring="roc_auc", cv=kfold
        )
        print(
            "Done cross validatiion. Validated AUC: {:.2f} (+/- {})".format(
                results.mean() * 100, results.std() * 100
            )
        )
        pass
