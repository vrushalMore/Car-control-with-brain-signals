import pandas as pd

def label_frequency(value):
    if value in range(0, 4):
        return 'D'
    elif value in range(4, 6):
        return 'T'
    elif value in range(6, 11):
        return 'TA'
    elif value in range(11, 21):
        return 'AB'
    elif value in range(21, 34):
        return 'B'
    elif value in range(33, 101):
        return 'G'
    else:
        return 'I'

def label_action_male(current, previous, after):
    if current in ['T']:
        if previous in ['T', 'TA', 'B'] or after in ['T', 'TA', 'B']:
            return 'R' 
    elif current in ['TA', 'AB']:
        if previous in ['T', 'TA', 'AB', 'B'] or after in ['T', 'TA', 'AB', 'B']:
            return 'L'
    elif current == 'TA':
        if previous in ['T', 'TA', 'B'] or after in ['T', 'TA', 'B']:
            return 'P' 
    elif current in ['D', 'G']:
        return 'W' 
    else:
        return 'F'

def label_action_female(current, previous, after):
    if current == 'TA':
        if previous in ['T', 'TA'] or after in ['T', 'TA']:
            return 'P'
    elif current in ['TA', 'AB']:
        if previous in ['T', 'TA', 'AB'] or after in ['T', 'TA', 'AB']:
            return 'L'
    elif current == 'TA':
        if previous in ['T', 'TA'] or after in ['T', 'TA']:
            return 'R'
    elif current in ['D', 'G']:
        return 'W'
    else:
        return 'F'

eeg_data = pd.read_csv('../dataset/data.csv')
eeg_data['frequency_label'] = eeg_data['value'].apply(label_frequency)

male_labels = []
female_labels = []

for index in range(len(eeg_data)):
    current = eeg_data.loc[index, 'frequency_label']
    previous = eeg_data.loc[index - 1, 'frequency_label'] if index > 0 else None
    after = eeg_data.loc[index + 1, 'frequency_label'] if index < len(eeg_data) - 1 else None

    male_labels.append(label_action_male(current, previous, after))
    female_labels.append(label_action_female(current, previous, after))

eeg_data['male_action_label'] = male_labels
eeg_data['female_action_label'] = female_labels

eeg_data[['timestamp', 'value', 'frequency_label', 'male_action_label']].to_csv('male.csv', index=False)
eeg_data[['timestamp', 'value', 'frequency_label', 'female_action_label']].to_csv('female.csv', index=False)

