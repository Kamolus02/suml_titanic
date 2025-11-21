import mlflow
from mlflow.models import infer_signature

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dsp6 = pd.read_csv("data/DSP_6_cleaned.csv")
print(dsp6)

X = dsp6.drop('Survived', axis=1)
y = dsp6['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

params_forest = {
    'n_estimators': 10,
    'random_state': 0
}

params_lreg = {
    'max_iter': 100,
    'random_state': 0,
    'solver': 'lbfgs'
}

# forest
forest = RandomForestClassifier(**params_forest)
forest.fit(X_train, y_train)
scores_forest = forest.score(X_train, y_train)
y_pred_1 = forest.predict(X_test)
acc_forest = accuracy_score(y_test, y_pred_1)
print(acc_forest)

# logistic reggresion
lreg = LogisticRegression(**params_lreg)
lreg.fit(X_train, y_train)
scores_lreg = lreg.score(X_train, y_train)
y_pred_2 = lreg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_2)
print(acc_log)

# decision tree classifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
tree_score = tree.score(X_train, y_train)
y_pred_3 = tree.predict(X_test)
acc_dec = accuracy_score(y_test, y_pred_3)
print(acc_dec)

mlflow.set_tracking_uri(uri = "http://localhost:8080")

mlflow.set_experiment("MLflow Titanic")

with mlflow.start_run(run_name="MLflow Titanic"):
    mlflow.log_params(params_forest)
    mlflow.log_params(params_lreg)
    mlflow.log_metric("acc_forest", acc_forest)
    mlflow.log_metric("acc_log", acc_log)
    mlflow.log_metric("acc_dec", acc_dec)

    mlflow.set_tag("Training Infor",
                   "Standardowe modele: losowe lasy, reresja logistyczna, drzewa decyzyjne dla danych Titanic")

    sign_forest = infer_signature(X_train, forest.predict(X_train))
    sign_lreg = infer_signature(X_train, lreg.predict(X_train))
    sign_dec = infer_signature(X_train, tree.predict(X_train))

    model_info_forest = mlflow.sklearn.log_model(
        sk_model=forest,
        artifact_path="titanic_model_forest",
        signature=sign_forest,
        input_example=X_train,
        registered_model_name="titanic_ml_forest"
    )

    model_info_lreg = mlflow.sklearn.log_model(
        sk_model=lreg,
        artifact_path="titanic_model_lreg",
        signature=sign_lreg,
        input_example=X_train,
        registered_model_name="titanic_ml_lreg"
    )

    model_info_tree = mlflow.sklearn.log_model(
        sk_model=tree,
        artifact_path="titanic_model_tree",
        signature=sign_dec,
        input_example=X_train,
        registered_model_name="titanic_ml_tree"
    )

loaded_model_forest = mlflow.pyfunc.load_model(model_info_forest.model_uri)
loaded_model_lreg = mlflow.pyfunc.load_model(model_info_lreg.model_uri)
loaded_model_tree = mlflow.pyfunc.load_model(model_info_tree.model_uri)

y_pred1 = loaded_model_forest.predict(X_test)
y_pred2 = loaded_model_lreg.predict(X_test)
y_pred3 = loaded_model_tree.predict(X_test)

titanic_acc = ["Random Forest", "Logistic Regression", "Decision Tree"]

result = pd.DataFrame()
result[titanic_acc[0]] = y_pred1
result[titanic_acc[1]] = y_pred2
result[titanic_acc[2]] = y_pred3