from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_predictions_with_best_model(df):
    male_action_enc = LabelEncoder()
    frequency_enc = LabelEncoder()

    df['male_action_label'] = male_action_enc.fit_transform(df['male_action_label'])
    df['frequency_label'] = frequency_enc.fit_transform(df['frequency_label'])

    X = df[['timestamp', 'value', 'frequency_label']]
    y = df['male_action_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X)
    decoded_predictions = male_action_enc.inverse_transform(predictions)
    
    return decoded_predictions.tolist()

