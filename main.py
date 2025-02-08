import pyautogui
import time
import pandas as pd
from prediction.male_prediction import get_predictions_with_best_model

data = pd.read_csv('dataset/male.csv')

def execute_action(data):
    decoded_predictions = get_predictions_with_best_model(data)
    
    for label in decoded_predictions:
        if label == 'P':
            key = 's'
        elif label == 'L':
            key = 'a'
        elif label == 'R':
            key = 'd'
        elif label in ('W', 'F'):
            key = 'w'
        else:
            continue

        pyautogui.keyDown(key)
        time.sleep(5)
        pyautogui.keyUp(key)

execute_action(data)

