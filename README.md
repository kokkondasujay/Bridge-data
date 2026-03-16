# 🌉 Bridge Condition Prediction using Machine Learning

An **end-to-end Machine Learning project** that predicts the structural condition of bridges using inspection and structural attributes.  
This project demonstrates the **complete ML pipeline from data preprocessing to model deployment**.

---

# 🚀 Live Demo

Try the deployed ML application:

👉 **HuggingFace Space**  
https://huggingface.co/spaces/sujay1234/bridge_condition

The application allows users to input bridge parameters and obtain predictions indicating whether the bridge is in **Good or Poor condition**.

---

# 📌 Project Overview

Bridge infrastructure is essential for transportation and economic development. Over time, bridges deteriorate due to aging, environmental factors, and heavy traffic loads.

Traditional bridge inspection methods are:

- Time-consuming
- Expensive
- Dependent on manual inspection

This project uses **Machine Learning models** to automatically predict bridge condition based on structural and operational features.

### Project Objectives

- Build multiple Machine Learning models
- Compare model performance
- Select the best performing model
- Deploy the model as an interactive web application

---

# 📂 Project Structure
Bridge-Condition-Prediction
│
├── data
│ └── bridge_dataset.csv
│
├── src
│ └── bridge_ml_pipeline.py
│
├── model
│ └── best_model.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md

---

# ⚙️ Machine Learning Workflow

The project follows a structured **Machine Learning pipeline**.

---

## 1️⃣ Data Cleaning

Initial preprocessing steps included:

- Handling missing values
- Removing duplicate records
- Checking feature distributions
- Performing Exploratory Data Analysis (EDA)

EDA visualizations used:

- Histograms
- Boxplots
- Countplots
- Correlation Heatmaps

These visualizations help understand relationships between bridge features and the target variable.

---

## 2️⃣ Data Splitting

The dataset was divided into:

- **Training Data → 80%**
- **Testing Data → 20%**

Training data is used to train the model, while testing data evaluates model performance on unseen data.

---

## 3️⃣ Data Preprocessing

Since the dataset contains both **numerical and categorical features**, preprocessing techniques were applied.

### StandardScaler

Used for **feature scaling** of numerical variables.

Standardization formula:

```
z = (x - μ) / σ
```

Where:

- **x** = original value  
- **μ** = mean  
- **σ** = standard deviation  

---

### OneHotEncoder

Used to convert **categorical variables into numerical format**.

Example:

```
Material Type

Steel
Concrete
Wood
```

After Encoding:

```
Steel  Concrete  Wood
1      0         0
0      1         0
0      0         1
```

---

### ColumnTransformer

Used to combine preprocessing steps for:

- numerical features
- categorical features

into a single preprocessing pipeline.

---

# 🤖 Machine Learning Models Used

Multiple **classification models** were trained and compared.

### Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

Each model represents a different machine learning approach.

| Model | Category |
|------|---------|
| Logistic Regression | Linear Model |
| KNN | Distance-Based Model |
| Decision Tree | Rule-Based Model |
| Random Forest | Ensemble Learning |

---

# 📊 Model Evaluation Metrics

Models were evaluated using classification metrics.

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```
Precision = TP / (TP + FP)
```

### Recall

```
Recall = TP / (TP + FN)
```

### F1 Score

```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

These metrics help measure the reliability of the classification model.

---

# 📈 Model Performance Comparison

| Model | Performance |
|------|------------|
| Logistic Regression | Moderate |
| KNN | Moderate |
| Decision Tree | Good |
| Random Forest | Best Performance |

Random Forest performed best due to its **ensemble learning capability**, which combines multiple decision trees to reduce overfitting and improve prediction stability.

---

# 🧠 Ensemble Methods

The project also explores ensemble techniques such as:

- Voting Classifier
- Stacking Classifier

Ensemble methods combine predictions from multiple models to produce more accurate and stable results.

---

# 💾 Model Serialization

The final trained model was saved using **Pickle**.

```
best_model.pkl
```

Model serialization allows the trained model to be reused without retraining.

---

# 🌐 Deployment

The trained model was deployed using **Gradio** and hosted on **HuggingFace Spaces**.

The deployed application allows users to:

1. Enter bridge parameters
2. Submit the input values
3. Receive real-time bridge condition predictions

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio
- HuggingFace Spaces
- Pickle

---

# 🔮 Future Improvements

Possible improvements include:

- Using larger real-world bridge inspection datasets
- Implementing advanced models like XGBoost
- Integrating IoT sensor data for real-time monitoring
- Developing dashboards for infrastructure monitoring

