import pandas as pd
import numpy as np
import random

# Set a random seed for reproducibility
np.random.seed(0)

# Generate time data
time = np.arange(0, 3000)

# Generate features V1 to V28
num_samples = 3000
num_features = 28
features = np.random.randn(num_samples, num_features)

# Generate Amount values between 1 and 500
amounts = np.random.uniform(1, 500, num_samples)

# Generate class labels (0 for normal, 1 for fraud)
fraud_indices = random.sample(range(num_samples), 20)  # Creating 20 fraud cases
class_labels = np.zeros(num_samples)
class_labels[fraud_indices] = 1

# Create the DataFrame
data = pd.DataFrame({
    "Time": time,
    "V1": features[:, 0],
    "V2": features[:, 1],
    "V3": features[:, 2],
    "V4": features[:, 3],
    "V5": features[:, 4],
    "V6": features[:, 5],
    "V7": features[:, 6],
    "V8": features[:, 7],
    "V9": features[:, 8],
    "V10": features[:, 9],
    "V11": features[:, 10],
    "V12": features[:, 11],
    "V13": features[:, 12],
    "V14": features[:, 13],
    "V15": features[:, 14],
    "V16": features[:, 15],
    "V17": features[:, 16],
    "V18": features[:, 17],
    "V19": features[:, 18],
    "V20": features[:, 19],
    "V21": features[:, 20],
    "V22": features[:, 21],
    "V23": features[:, 22],
    "V24": features[:, 23],
    "V25": features[:, 24],
    "V26": features[:, 25],
    "V27": features[:, 26],
    "V28": features[:, 27],
    "Amount": amounts,
    "Class": class_labels
})

# Print the first few rows of the dataset
print(data.head())
# Assuming you already have the 'data' DataFrame from the previous code

# Specify the filename
csv_filename = 'credit_card_data.csv'

# Save the DataFrame to a CSV file
data.to_csv(csv_filename, index=False)

print(f'Data saved to {csv_filename}')
