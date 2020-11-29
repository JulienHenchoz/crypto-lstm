import sys
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

from utils.dataset import Dataset

file_name = sys.argv[1]

dataset = Dataset()
df = pd.read_csv(file_name, index_col=None)
df = df.sort_values('datetime')

df = df.dropna()
df = df[0:1000]

# Number of features for prediction
features_count = df.shape[1]

# Make predictions using the reference model
predictions_reference, y = dataset.predict(df, './models/reference_model', True)
print('Mean error reference : ' + str(dataset.mean_error(np.array(predictions_reference), y)))
success_rate_reference = dataset.get_trend_success_rate(predictions_reference, y, df)
print ('Trend prediction success for reference : ' + str(success_rate_reference) + '%')

# Make predictions using the latest model if available
try:
    has_latest_model = True
    predictions_latest, _ = dataset.predict(df, './models/latest', True)
    print('Mean error latest : ' + str(dataset.mean_error(np.array(predictions_latest), y)))
    success_rate_latest = dataset.get_trend_success_rate(predictions_latest, y, df)
    print('Trend prediction success for latest : ' + str(success_rate_latest) + '%')
except:
    has_latest_model = False
    predictions_latest = None
    pass

figure(num=None, figsize=(24, 10), dpi=80, facecolor='w', edgecolor='k')

ax = sns.lineplot(x=predictions_reference.index, y=y, label="Test Data", color='blue')
ax = sns.lineplot(x=predictions_reference.index, y=predictions_reference[0], label="Prediction reference", color='gray')

if has_latest_model:
    ax = sns.lineplot(x=predictions_reference.index, y=predictions_latest[0], label="Prediction latest", color='red')

ax.set_title('Price', size = 14, fontweight='bold')
ax.set_xlabel("Hours", size = 14)
ax.set_ylabel("Cost", size = 14)
ax.set_xticklabels('', size=10)

plt.show()
