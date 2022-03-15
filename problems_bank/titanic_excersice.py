import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

Pclass_ix, Sex_ix, Age_ix = 1, 3, 4

class PclassSexAgeTransformer(BaseEstimator, TransformerMixin):
    """ Debe recibir X que no contenga la variable objetivo Survived """
    def __init__(self, only_three_features=True):
        self.only_three_features = only_three_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X_arr = X.values
        elif type(X) == np.ndarray:
            X_arr = X

        titanic_3 = X_arr[:, [Pclass_ix, Sex_ix, Age_ix]]
        titanic_num = titanic_3[:, [0, 2]] # just Pclass and Age
        imputer = SimpleImputer()
        X_num = imputer.fit_transform(titanic_num)
        titanic_cat = titanic_3[:, [1]] # just Sex
        encoder = OneHotEncoder()
        X_cat = encoder.fit_transform(titanic_cat)
        return np.hstack([X_num, X_cat.todense()])

# 2.

titanic = pd.read_csv("train.csv")

titanic_labels = titanic['Survived']
titanic = titanic.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(titanic, titanic_labels)

psa_trans = PclassSexAgeTransformer()

titanic_trian_prepared = psa_trans.fit_transform(X_train)
titanic_trian_prepared_df = pd.DataFrame(titanic_trian_prepared,
                                        columns=["Pclass", "Age", "Sex1", "Sex2"],
                                        index=X_train.index)

titatic_test_prepared = psa_trans.transform(X_test)

# models
reg_log_clf = LogisticRegression()
r_forest_clf = RandomForestClassifier()

reg_log_clf.fit(titanic_trian_prepared, y_train)
y_pred_reg_log = reg_log_clf.predict(titatic_test_prepared)
print('Score with LogisticRegression: ')
print(reg_log_clf.score(titatic_test_prepared, y_test))
cm_reg_log = confusion_matrix(y_test, y_pred_reg_log)
print(cm_reg_log)


r_forest_clf.fit(titanic_trian_prepared, y_train)
y_pred_r_forest = r_forest_clf.predict(titatic_test_prepared)
print("Score with RandomForestClassifier: ")
print(r_forest_clf.score(titatic_test_prepared, y_test))
cm_r_forest = confusion_matrix(y_test, y_pred_r_forest)
print(cm_r_forest)


# 3. GridSearchCV # Queda pendiente la lectura sobre classification_report()

param_grid_reglog = [
            {"penalty":['none']},
            {"penalty":['l2'], "C":[1.0, 0.1, 0.01]}
]

param_grid_r_forest = {"n_estimators":[8, 10, 12], "max_depth":[2,4]}

reglog_gs = GridSearchCV(reg_log_clf,
                        param_grid_reglog,
                        scoring="precision",
                        cv=4, n_jobs=-1).fit(titanic_trian_prepared, y_train)
randomfo_gs = GridSearchCV(r_forest_clf,
                        param_grid_r_forest,
                        scoring="accuracy",
                        cv=4, n_jobs=-1).fit(titanic_trian_prepared, y_train)

print("Mejor modelo tipo LogisticRegression y su respectivo puntaje: ")
print(reglog_gs.best_params_, reglog_gs.best_score_)

print("Mejor modelo tipo RandomForestClassifier y su respectivo puntaje: ")
print(randomfo_gs.best_params_, randomfo_gs.best_score_)


# 4. Pipeline y GridSearchCV sobre el pipeline, incluyendo los dos modelos
pipe = Pipeline([
                ("scaler", StandardScaler()), # para el futuro seria bueno incluir PclassSexAgeTransformer
                ("classifier", LogisticRegression())
])

pipe_param_grid = [
        {"classifier": [LogisticRegression()],
         "classifier__penalty":['l2'],
         "classifier__C":[1.0, 0.1, 0.01, 0.001]},
         {"classifier": [LogisticRegression()],
          "classifier__penalty":['none']},
        {"classifier": [RandomForestClassifier()],
         "classifier__n_estimators":[6,8,10,12],
         "classifier__max_depth":[2,4,6]}
]

pipe_gs = GridSearchCV(pipe,
                        pipe_param_grid,
                        cv=5,
                        scoring="accuracy",
                        n_jobs=-1).fit(titanic_trian_prepared, y_train)

params = pipe_gs.best_params_
print("Best Classifier: ")

print(confusion_matrix(y_test, pipe_gs.predict(titatic_test_prepared)))
