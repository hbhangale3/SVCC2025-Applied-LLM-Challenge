import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the Kyoto dataset
kyoto_df = pd.read_csv('Kyoto.csv')

# Preview structure
print("Dataset Shape:", kyoto_df.shape)
print("Columns:", kyoto_df.columns.tolist())
print(kyoto_df.head(2))

# Encode categorical columns
kyoto_df['protocol_type_n'] = kyoto_df['protocol_type'].astype('category').cat.codes
kyoto_df['Service_n'] = kyoto_df['Service'].astype('category').cat.codes
kyoto_df['Flag_n'] = kyoto_df['Flag'].astype('category').cat.codes

# Map Label column (-1: normal, 1: attack)
kyoto_df['label_n'] = kyoto_df['Label'].map({-1: 0, 1: 1})

# Keep only necessary features
features = [
    'Duration', 'Service_n', 'src_bytes', 'dst_bytes', 'Count', 'Same srv rate',
    'Serror rate', 'Srv serror rate', 'Dst host count', 'Dst host srv count',
    'Dst host same src port rate', 'Dst host serror rate', 'Dst host srv serror rate',
    'Flag_n', 'protocol_type_n'
]

# Drop rows where label is missing
kyoto_df_clean = kyoto_df.dropna(subset=['label_n'])

# Fill missing feature values with 0
kyoto_df_clean[features] = kyoto_df_clean[features].fillna(0)

# Final check on feature integrity
print("Cleaned dataset shape:", kyoto_df_clean.shape)

# Split data
eighty_cut_n = math.ceil(len(kyoto_df_clean) * 0.8)
train_x = kyoto_df_clean.iloc[:eighty_cut_n][features]
train_y = kyoto_df_clean.iloc[:eighty_cut_n]['label_n']
test_x = kyoto_df_clean.iloc[eighty_cut_n:][features]
test_y = kyoto_df_clean.iloc[eighty_cut_n:]['label_n']

print("Train shape:", train_x.shape, train_y.shape)
print("Test shape:", test_x.shape, test_y.shape)

# Ensure there is enough data
if train_x.empty or test_x.empty:
    print("Error: Training or testing set is empty.")
else:
    # GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    best_rf = grid_search.best_estimator_

    # RFE
    selector = RFE(estimator=best_rf, n_features_to_select=5)
    selector.fit(train_x, train_y)
    selected_features = train_x.columns[selector.support_]
    print("Selected Features:", selected_features.tolist())

    # Final training
    train_x_sel = train_x[selected_features]
    test_x_sel = test_x[selected_features]
    best_rf.fit(train_x_sel, train_y)
    pred_test_y = best_rf.predict(test_x_sel)

    # Evaluation
    acc = accuracy_score(test_y, pred_test_y)
    conf = confusion_matrix(test_y, pred_test_y)
    tn, fp, fn, tp = conf.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("Confusion Matrix:\n", conf)

    # ----- PLOTS -----

    # Class distribution
    plt.figure(figsize=(6, 4))
    kyoto_df_clean['label_n'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title("Class Distribution")
    plt.xticks([0, 1], ['Normal', 'Anomaly'], rotation=0)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=['Normal', 'Anomaly'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    # Feature importances
    importances = pd.Series(best_rf.feature_importances_, index=selected_features).sort_values(ascending=True)
    plt.figure(figsize=(6, 4))
    importances.plot(kind='barh', color='skyblue')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

