import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE

# Load the data
train_data = pd.read_csv('Train_data.csv')
test_data = pd.read_csv('Test_data.csv')

# Combine train and test for unified processing
data_df = pd.concat([train_data, test_data], ignore_index=True)

# Feature Engineering
data_df['protocol_type_n'] = data_df['protocol_type'].map({'tcp': 0, 'udp': 1, 'icmp': 2})
data_df['service_n'] = data_df['service'].astype('category').cat.codes
data_df['flag_n'] = data_df['flag'].astype('category').cat.codes
data_df['class_n'] = data_df['class'].map({'normal': 0, 'anomaly': 1})

# Define features to use
features = ['duration', 'protocol_type_n', 'service_n', 'flag_n', 'dst_bytes',
            'dst_host_srv_rerror_rate', 'root_shell', 'dst_host_same_src_port_rate',
            'serror_rate', 'dst_host_serror_rate']

# Drop any rows with NaNs in features/label
data_df_clean = data_df.dropna(subset=features + ['class_n']).reset_index(drop=True)

# Split data
eighty_cut_n = math.ceil(data_df_clean.shape[0] * 0.8)
train_x = data_df_clean.loc[:eighty_cut_n, features]
train_y = data_df_clean.loc[:eighty_cut_n, 'class_n']
test_x = data_df_clean.loc[eighty_cut_n:, features]
test_y = data_df_clean.loc[eighty_cut_n:, 'class_n']

# GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(train_x, train_y)
best_rf = grid_search.best_estimator_

# Feature selection using RFE
selector = RFE(estimator=best_rf, n_features_to_select=5)
selector.fit(train_x, train_y)

# Get top selected features
selected_features = train_x.columns[selector.support_]
print("Selected Features:", selected_features.tolist())

# Retrain using selected features
train_x_selected = train_x[selected_features]
test_x_selected = test_x[selected_features]
best_rf.fit(train_x_selected, train_y)
pred_test_y = best_rf.predict(test_x_selected)

# Evaluate
accuracy = accuracy_score(test_y, pred_test_y)
conf_matrix = confusion_matrix(test_y, pred_test_y)
tn, fp, fn, tp = conf_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"\n Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# ------------------- VISUALIZATION -------------------

# 1. Class distribution
plt.figure(figsize=(6, 4))
data_df_clean['class_n'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("KDD Dataset: Class Distribution")
plt.xticks([0, 1], ['Normal', 'Anomaly'], rotation=0)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 2. Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Anomaly'])
disp.plot(cmap='Blues')
plt.title("KDD Dataset: Confusion Matrix")
plt.grid(False)
plt.show()

# 3. Feature Importance Plot
feature_importances = pd.Series(best_rf.feature_importances_, index=selected_features).sort_values(ascending=True)
plt.figure(figsize=(6, 4))
feature_importances.plot(kind='barh', color='skyblue')
plt.title("Top 5 Feature Importances (RFE)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

