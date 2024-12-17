import pandas as pd
import numpy as np

data_path = './data/modified_data.csv'
df = pd.read_csv(data_path)

from automate.null_value_remover import Null_Value_Remover

df = Null_Value_Remover.remove_null_value(df,df['timestamp'])

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

def calculateStandardDeviation(value):
    return np.std(value)

stdDevList = []

for i in range(len(df)):
    stdDev = calculateStandardDeviation(df['value'].iloc[:i+1])
    stdDevList.append(stdDev)

df['StandardDeviation'] = stdDevList
df['stdDev'] = df['StandardDeviation']

df['signalPower'] = df['value'] ** 2
df['RMS'] = np.sqrt(df['value'] ** 2)
df['bandCategory'] = df['bandCategory'].map({'alpha': 0, 'beta': 1, 'delta': 2, 'theta': 3, 'gamma': 4})

df.dropna(subset=['stdDev', 'state', 'RMS'], inplace=True)

df['state'] = df['state'].map({'A': 1, 'R': 0})

df = Null_Value_Remover.remove_null_value(df, categorial_or_timestamp_columns=['bandCategory'])

new_df = df.copy()

new_df.to_csv('modified_data.csv',index=False)