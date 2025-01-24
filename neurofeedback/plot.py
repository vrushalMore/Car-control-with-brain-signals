import matplotlib.pyplot as plt
import pandas as pd
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  
file_path = os.path.join(current_dir, '../dataset/data.csv')
eeg_data = pd.read_csv(file_path)

def classify_frequency(band_data, line, ax):
    for i in range(0, len(band_data)):
        frequency = band_data.iloc[i, 1]
        
        if frequency in range(0, 3):
            band = 'delta'
        elif frequency in range(4, 7):
            band = 'theta'
        elif frequency in range(8, 12):
            if frequency in range(8, 10):
                band = 'Lower alpha'
            else:
                band = 'Upper alpha'
        elif frequency in range(13, 30):
            if frequency in range(13, 15):
                band = 'Sensorimotor rhythm'
            elif frequency in range(16, 20):
                band = 'Lower beta'
            else:
                band = 'Higher beta'
        elif frequency in range(33, 100):
            band = 'gamma'
        else:
            band = 'Invalid'
        
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Current Band: {band}", end="\r")
        
        live_data = band_data.iloc[:i+1, 1]
        line.set_ydata(live_data)
        line.set_xdata(range(len(live_data)))
        
        ax.set_xlim(0, len(live_data))
        ax.set_ylim(min(live_data)-5, max(live_data)+5)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)
        
        if i < len(band_data) - 1:
            time_interval = 1
            time.sleep(time_interval)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='EEG Data', color='b')
ax.set_title('Live EEG Data Plot')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.legend()

classify_frequency(eeg_data, line, ax)

plt.ioff()
plt.show()
