import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('eeg_data.csv')

df.info()
df.head()
df.describe()
print("Missing val:\n", df.isnull().sum())

plt.hist(df['value'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of EEG Signal Values')
plt.xlabel('EEG Signal Value')
plt.ylabel('Frequency')
plt.show()

band_counts = df['bandCategory'].value_counts()
plt.bar(band_counts.index, band_counts.values, color='orange')
plt.title('Count of Each Band Category')
plt.xlabel('Band Category')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(df.corr(), cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Heatmap')
plt.show()

plt.boxplot([df[df['bandCategory'] == category]['value'] for category in df['bandCategory'].unique()], labels=df['bandCategory'].unique())
plt.title('Boxplot of EEG Signal Values by Band Category')
plt.xlabel('Band Category')
plt.ylabel('EEG Signal Value')
plt.show()
