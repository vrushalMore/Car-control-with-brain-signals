import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pyautogui
import time

df = pd.read_csv('modified_data.csv')

X = df[['stdDev', 'signalPower', 'RMS', 'bandCategory']]
Y = df['state']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

best_model = None
best_accuracy = 0

for model_name, model in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(x_train, y_train)
    accuracy = pipeline.score(x_test, y_test)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_name

pipeline = Pipeline([("scaler", StandardScaler()), ("model", models[best_model])])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

for state in y_pred:
    if state == 1:
        pyautogui.press('w')
    else:
        pyautogui.press('space')
    time.sleep(0.5)
