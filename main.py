import pyautogui
import time
import pandas as pd
from prediction.male_prediction import get_predictions_with_best_model

data = pd.read_csv('dataset/male.csv',nrows = 100)

def execute_action(data):
    decoded_predictions = get_predictions_with_best_model(data)
    
    for label in decoded_predictions:
        if label == 'P':
            pyautogui.press('space')
        elif label == 'L':
            pyautogui.press('a')
        elif label == 'R':
            pyautogui.press('d')
        elif label == 'W':
            pyautogui.press('w')
        elif label == 'F':
            pyautogui.press('s')
        
        time.sleep(0.5)



execute_action(data)

