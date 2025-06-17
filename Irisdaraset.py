# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
# Loading the dataset
# Obtener la ruta absoluta de la imagen
iris_path = os.path.abspath("Data/IrisSpecies.csv")

# Read the CSV file into a DataFrame
iris = pd.read_csv(iris_path)

# Display first 5 rows
print("First 5 rows of the dataset:")
print(iris.head())  # Note: using df.head() instead of iris.head()

# Dataset Info
print("\nDataset Information:")
iris.info()

# Checking for missing values
print("\nMissing values in the dataset:")
print(iris.isnull().sum())

# Basic Statistics
print("\nDescriptive Statistics:")
print(iris.describe())

# Class distribution
print("\nClass Distribution:")
print(iris['Species'].value_counts())

# Correlation Matrix
print("\nCorrelation Matrix:")
print(iris.corr(numeric_only=True))

# Heatmap for correlation
plt.figure(figsize=(8,5))
sns.heatmap(iris.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot (scatterplot matrix)
sns.pairplot(iris, hue='Species')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Boxplots
plt.figure(figsize=(15,10))
for i, column in enumerate(iris.columns[:-1], 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='Species', y=column, data=iris)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# Violin Plots
plt.figure(figsize=(15,10))
for i, column in enumerate(iris.columns[:-1], 1):
    plt.subplot(3, 2, i)
    sns.violinplot(x='Species', y=column, data=iris)
    plt.title(f'Violin plot of {column}')
plt.tight_layout()
plt.show()

# Scatter Plot 
# For example, sepal length vs sepal width
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Scatter Plot 
# For example, sepal length vs sepal width
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=iris)
plt.title('Petal Length vs Petal Width')
plt.show()
     
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

Y=iris.loc[:, 'Species']
X=iris.drop('Species', axis=1)

X_corr=X.corr()
X_corr

Y.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

X_train = X_train.drop(columns=['Id'])  # Adjust column name
X_test = X_test.drop(columns=['Id'])


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline: Escalado + Modelo
model = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar datos
    ('knn', KNeighborsClassifier(n_neighbors=3))  # KNN con 3 vecinos
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline: Escalado + Modelo
model = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar datos
    ('knn', KNeighborsClassifier(n_neighbors=3))  # KNN con 3 vecinos
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)





from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, Y, cv=5)  # 5-fold CV
print(f"Accuracy promedio: {scores.mean():.2f}")

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')

from xgboost import XGBClassifier

# Modelo XGBoost (ajustado para multiclase)
xgb_model = XGBClassifier(
    objective='multi:softmax',  # Para multiclase
    num_class=3,               # Número de clases en Iris
    n_estimators=100,          # Número de árboles
    learning_rate=0.1,         # Tasa de aprendizaje
    random_state=42
)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_model.fit(X_train, y_train_encoded)
y_pred_encoded = xgb_model.predict(X_test)

# Decode for human-readable results
y_pred_original = le.inverse_transform(y_pred_encoded)

# Now you can use either version for metrics
print(classification_report(y_test, y_pred_original))  # Most readable

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_original), annot=True, fmt='d', cmap='Oranges', xticklabels=iris['Species'].unique(), yticklabels=iris['Species'].unique())
plt.title("Matriz de Confusión - XGBoost")
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 1. Define the SVM model
svm_model = SVC(random_state=42)  # Create the model first

# 2. Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# 3. Now perform GridSearchCV
grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=4, verbose=2)
grid_svm.fit(X_train, y_train_encoded)  # Use encoded y_train if needed

# Best parameters and score
print("Best parameters:", grid_svm.best_params_)
print("Best score:", grid_svm.best_score_)

from xgboost import plot_importance
plot_importance(xgb_model)
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the model
svm_model = SVC(random_state=42)

# CORRECT parameter grid (only valid SVC parameters)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],            # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],     # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly']  # Kernel type
}

# Create and run GridSearchCV
grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, verbose=2)
grid_svm.fit(X_train, y_train)  # Use y_train_encoded if you encoded labels

# Show results
print("Best parameters:", grid_svm.best_params_)
print("Best score:", grid_svm.best_score_)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Precisión Random Forest Clasificación: {}'.format(clf.score(X_train, y_train)))


from sklearn.tree import DecisionTreeClassifier

#Modelo de Árboles de Decisión Clasificación
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Árboles de Decisión Clasificación: {}'.format(algoritmo.score(X_train, y_train)))

# Entrenar XGBoost
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

# Convert string labels to numerical codes
from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  # Fit on train data
y_test_encoded = label_encoder.transform(y_test)        # Apply same encoding to test data

# Train XGBoost
xgb_model.fit(X_train, y_train_encoded)

# Predict (returns 0, 1, 2)
y_pred_encoded = xgb_model.predict(X_test)

# ✅ Correct way to decode predictions back to original labels
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)


# Evaluate
print(classification_report(y_test_encoded, y_pred_encoded))
print(classification_report(y_test, y_pred_original))

# Evaluarfrom sklearn.metrics import accuracy_score

# Using encoded labels (0,1,2)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

# Using original labels ('Iris-setosa', etc.)
accuracy_original = accuracy_score(y_test, y_pred_original)

print(f"Accuracy (Encoded): {accuracy:.4f}")
print(f"Accuracy (Original Labels): {accuracy_original:.4f}")


import pickle

# Guardar el modelo
with open('xgb_modelo_iris.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

print("✅ Modelo guardado como 'xgb_modelo_iris.pkl'")


import joblib

# Guardar
joblib.dump(clf, 'randomforest.joblib')



# Guardar el LabelEncoder (para convertir strings a números)
label_encoder = LabelEncoder()
label_encoder.fit(Y)  # Ajustar con todas las clases originales
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Number of features in training:", X_train.shape[1])

# Example prediction
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Must match training format
print("Prediction:", xgb_model.predict(new_sample))

new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Must match training format
print("Prediction:", clf.predict(new_sample))
 
