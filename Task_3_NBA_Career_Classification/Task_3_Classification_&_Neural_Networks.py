#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# In[17]:


# Load the dataset
data = pd.read_csv('nba_rookie_data.csv')

# Display the first few rows and the summary to understand the dataset
print(data.head())
print(data.info())


# In[18]:


# Data Preprocessing
# Drop 'Name' column and any rows with NaN values
data.drop(columns=['Name'], inplace=True)
data.dropna(inplace=True)

# Features and target variable
X = data.drop('TARGET_5Yrs', axis=1)  # Use 'TARGET_5Yrs' as the target column
y = data['TARGET_5Yrs']

# Encode categorical variables if necessary (uncomment if needed)
# X = pd.get_dummies(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[19]:


# Model Training and Predictions

## 1. Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

## 2. Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

## 3. Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

## 4. Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

## 5. Neural Network Classifier
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
nn_model.add(Dense(32, activation='relu'))  # Hidden layer
nn_model.add(Dense(1, activation='sigmoid'))  # Output layer

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model

# Fit the model
nn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)  # Adjust epochs and batch_size as needed
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")  # Predict using threshold


# In[20]:


# Evaluation of models
print("Logistic Regression Results:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("Gaussian Naive Bayes Results:")
print(confusion_matrix(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))

print("Decision Tree Results:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

print("Neural Network Results:")
print(confusion_matrix(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))


# In[23]:


# Confusion Matrix Plotting Function
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Not Successful', 'Successful'], 
                yticklabels=['Not Successful', 'Successful'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    file_name = f'{model_name.replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(file_name)
    plt.show()

# Confusion Matrix for each model
models = {
    "Logistic Regression": y_pred_log,
    "Gaussian Naive Bayes": y_pred_gnb,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Neural Network": y_pred_nn
}

for model_name, y_pred in models.items():
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)


# In[22]:


# Plot ROC Curve for Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_log)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# ## Author: Ajakaiye, Oluwadamilola Oreofe
# ## Data Science Consultant
# ### Date: 21/10/2024

# In[ ]:





# In[ ]:





# In[ ]:




