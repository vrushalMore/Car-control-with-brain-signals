import pandas as pd
import numpy as np
import pyautogui
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

df = pd.read_csv('eeg_data.csv')

def categorizeFrequency(value):
    if 0.5 <= value < 4: return 'delta'
    elif 4 <= value < 8: return 'theta'
    elif 8 <= value < 13: return 'alpha'
    elif 13 <= value < 30: return 'beta'
    elif 30 <= value < 100: return 'gamma'
    else: return 'notNeeded'

df['bandCategory'] = df['value'].apply(categorizeFrequency)

def bandpassNotchFilter(df):
    df = df[(df['bandCategory'] == 'alpha') | (df['bandCategory'] == 'beta') | (~df['value'].between(50, 60))]
    return df

df = bandpassNotchFilter(df)

alphaSum = 0
betaSum = 0

for i in range(len(df)):
    if df.iloc[i]['bandCategory'] == 'alpha':
        alphaSum += df.iloc[i]['value'] ** 2
    elif df.iloc[i]['bandCategory'] == 'beta':
        betaSum += df.iloc[i]['value'] ** 2

df['alphaBetaRatio'] = None

for i in range(len(df)):
    if df.iloc[i]['bandCategory'] == 'alpha':
        alphaPower = df.iloc[i]['value'] ** 2
        if betaSum != 0:
            df.at[i, 'alphaBetaRatio'] = alphaPower / betaSum
    elif df.iloc[i]['bandCategory'] == 'beta':
        betaPower = df.iloc[i]['value'] ** 2
        if alphaSum != 0:
            df.at[i, 'alphaBetaRatio'] = alphaSum / betaPower

def stateCategorization():
    stateList = []
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['alphaBetaRatio']):
            stateList.append(None)
        elif df.iloc[i]['alphaBetaRatio'] > 1:
            stateList.append('A')
        else:
            stateList.append('R')
    df['state'] = stateList

stateCategorization()

df['StandardDeviation'] = df['value'].expanding().std()
df['signalPower'] = df['value'] ** 2
df['RMS'] = np.sqrt(df['value'] ** 2)
df['frequencyDrop'] = df['value'].diff()
df['bandCategory'] = df['bandCategory'].map({'alpha': 0, 'beta': 1, 'delta': 2, 'theta': 3, 'gamma': 4})
df.dropna(inplace=True)
df['state'] = df['state'].map({'A': 1, 'R': 0})

feature_columns = [
    'StandardDeviation', 'signalPower', 'RMS', 'bandCategory', 'frequencyDrop'
]

X = df[feature_columns]
Y = df['state']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

bestModels = []
bestAccuracy = 0

for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    with open('modelResults.txt', 'a') as f:
        f.write(f"Model: {name}, Accuracy: {100 * accuracy:.2f}%\n")
    
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestModels = [name]
    elif accuracy == bestAccuracy:
        bestModels.append(name)

with open('modelResults.txt', 'a') as f:
    f.write(f"\nBest Model(s): {', '.join(bestModels)} with Accuracy: {100 * bestAccuracy:.2f}%\n")



for i in range(len(df)):
    if df.iloc[i]['state'] == 1:
        pyautogui.press('w')
    elif df.iloc[i]['state'] == 0:
        pyautogui.press('s')
