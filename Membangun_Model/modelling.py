import pandas as pd
import os 
import mlflow 
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "whatsapp_review_preprocessing", "train_processed.csv")
    test_path = os.path.join(base_path, "whatsapp_review_preprocessing", "test_processed.csv") 

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) 

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1] 

    return X_train, y_train, X_test, y_test 

def train():
    mlflow.sklearn.autolog() 
    mlflow.set_experiment("Eksperimen Basic")
    with mlflow.start_run(run_name="Basic_Random_Forest"):
        X_train, y_train, X_test, y_test = load_data() 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score RF: {acc}")
    with mlflow.start_run(run_name="Basic_Decision_Tree"):
        X_train, y_train, X_test, y_test = load_data() 
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score DT: {acc}")
    with mlflow.start_run(run_name="Basic_Naive_Bayes"):
        X_train, y_train, X_test, y_test = load_data() 
        model = BernoulliNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score NB: {acc}")
    with mlflow.start_run(run_name="Basic_Logistic_Regression"):
        X_train, y_train, X_test, y_test = load_data() 
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score LR: {acc}") 
        
if __name__ == "__main__":
    train()
    