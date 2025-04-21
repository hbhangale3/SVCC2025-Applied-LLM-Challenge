
Intrusion Detection System using Random Forest on KDD & Kyoto Datasets
=====================================================================

This project implements a classical Machine Learning pipeline to detect anomalies in network traffic using two widely used datasets: KDD99 and Kyoto 2006+. It showcases how to prepare, tune, evaluate, and visualize a Random Forest-based binary classifier for intrusion detection.

---------------------------------------------------------------------
Requirements
---------------------------------------------------------------------
Before running the scripts, install the following dependencies:

pip install pandas numpy matplotlib scikit-learn

---------------------------------------------------------------------
File Structure
---------------------------------------------------------------------
.
├── kdd_pipeline.py         # KDD dataset processing and modeling
├── kyoto_pipeline.py       # Kyoto dataset processing and modeling
├── Train_data.csv          # KDD training data
├── Test_data.csv           # KDD test data
├── Kyoto.csv               # Kyoto dataset

---------------------------------------------------------------------
How to Run
---------------------------------------------------------------------

For KDD Dataset
python kdd_pipeline.py
> Requires: Train_data.csv, Test_data.csv

For Kyoto Dataset
python kyoto_pipeline.py
> Requires: Kyoto.csv

---------------------------------------------------------------------
Pipeline Breakdown (Applicable to Both Scripts)
---------------------------------------------------------------------

1. Import Libraries
2. Load Data
3. Feature Engineering: Encode categorical features & map class labels
4. Clean Data: Drop or fill NaN values
5. Train-Test Split (manual 80/20)
6. Hyperparameter Tuning with GridSearchCV
7. Feature Selection using RFE (top 5 features)
8. Model Training & Prediction
9. Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix
10. Visualization: Class distribution, Confusion matrix, Feature importance

---------------------------------------------------------------------
KDD Dataset Notes (kdd_pipeline.py)
---------------------------------------------------------------------

- Data Source: KDD Cup 1999 (cleaned version)
- Features Mapped:
  - protocol_type → protocol_type_n
  - service → service_n
  - flag → flag_n
  - class → class_n (0 for normal, 1 for anomaly)

Selected Feature Pool:
['duration', 'protocol_type_n', 'service_n', 'flag_n', 'dst_bytes',
 'dst_host_srv_rerror_rate', 'root_shell', 'dst_host_same_src_port_rate',
 'serror_rate', 'dst_host_serror_rate']

---------------------------------------------------------------------
Kyoto Dataset Notes (kyoto_pipeline.py)
---------------------------------------------------------------------

- Data Source: Kyoto 2006+ IDS Logs
- Features Mapped:
  - protocol_type → protocol_type_n
  - Service → Service_n
  - Flag → Flag_n
  - Label → label_n (-1 = normal, 1 = anomaly)

Selected Feature Pool:
['Duration', 'Service_n', 'src_bytes', 'dst_bytes', 'Count',
 'Same srv rate', 'Serror rate', 'Srv serror rate', 'Dst host count',
 'Dst host srv count', 'Dst host same src port rate', 'Dst host serror rate',
 'Dst host srv serror rate', 'Flag_n', 'protocol_type_n']

---------------------------------------------------------------------
Output Visualizations
---------------------------------------------------------------------
Both scripts generate:
- Class Distribution Plot
- Confusion Matrix Plot
- Feature Importance Plot (Top 5 from RFE)

---------------------------------------------------------------------
Observations
---------------------------------------------------------------------

| Metric     | KDD Dataset (Synthetic) | Kyoto Dataset (Real-world) |
|------------|--------------------------|-----------------------------|
| Accuracy   | High (~95%–99%)          | Moderate (~75%–90%)        |
| Precision  | High                     | Lower due to imbalance      |
| Recall     | High                     | More affected by noise      |
| Dataset    | Clean & balanced         | Noisy & imbalanced          |

---------------------------------------------------------------------
References
---------------------------------------------------------------------

- KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- Kyoto 2006+: http://www.takakura.com/Kyoto_data/
- scikit-learn: https://scikit-learn.org/stable/


---------------------------------------------------------------------
Modifications & Techniques Used
---------------------------------------------------------------------

Initially, the first approach was to **use all available features** from the dataset and run the model without any feature selection. The goal was to observe baseline values for:
- **Accuracy**
- **Precision**
- **Recall**

**Observation**:  
Accuracy and precision stayed fairly stable, but **recall dipped slightly**, indicating the model missed detecting some anomalies.

---

### Hyperparameter Tuning with GridSearchCV

Next, we applied **GridSearchCV** to:
- Search over a range of values for `n_estimators`, `max_depth`, and `min_samples_split` for `RandomForestClassifier`.
- Use **cross-validation** to ensure the best model generalizes well on unseen data.

**Result**:  
Improved performance with optimized parameters.

---

### Manual Feature Dropping

To test the effect of certain features, we **manually removed** features like:
- `root_shell`
- `serror_rate`

**Result**:  
Model remained performant but showed slight variations, suggesting these features had limited influence.

---

### Automated Feature Selection Techniques

#### 1. **SelectFromModel**
- Assigns an importance score (coefficient) to each feature.
- Selects features that **exceed a specified threshold**.
- Filters out low-impact features.

**Result**:  
Good precision and accuracy. Recall varied depending on the feature threshold.

#### 2. **RFE (Recursive Feature Elimination)**
- Starts with all features.
- Recursively trains the model and **removes the least important feature**.
- Continues until the desired number of features is reached (e.g., top 5).

🔍 **Result**:  
Outperformed baseline model. Outputted an optimal subset of features that improved overall classification performance, especially recall for anomalies.

---

**Conclusion**:
Through iterative refinement — GridSearchCV tuning, selective feature dropping, and automated selection techniques like RFE — the final pipeline achieves **better generalization**, especially for **detecting anomalies** in imbalanced datasets like KDD and Kyoto.

