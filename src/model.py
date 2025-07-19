from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_models(X_train, y_train):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    return xgb, logreg

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return report

