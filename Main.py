import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
stroke_df = pd.read_csv("stroke_dataset.csv")

# Select 10 features for input
feature_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
X = stroke_df[feature_cols]
y = stroke_df['stroke']

# Impute missing values
imp = SimpleImputer(strategy="mean")
X[['bmi']] = imp.fit_transform(X[['bmi']])

# Convert categorical variables to numerical variables
X = pd.get_dummies(X, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Make predictions on new data
new_data = pd.DataFrame({'age': [65], 'hypertension': [1], 'heart_disease': [0], 'avg_glucose_level': [60], 'bmi': [22], 'gender_Female': [1], 'gender_Male': [0], 'gender_Other': [0], 'ever_married_No': [0], 'ever_married_Yes': [1], 'work_type_Govt_job': [0], 'work_type_Never_worked': [0], 'work_type_Private': [1], 'work_type_Self-employed': [0], 'work_type_children': [0], 'Residence_type_Rural': [0], 'Residence_type_Urban': [1], 'smoking_status_Unknown': [0], 'smoking_status_formerly smoked': [0], 'smoking_status_never smoked': [1], 'smoking_status_smokes': [0]})
new_data = pd.get_dummies(new_data)
new_data = selector.transform(new_data)
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)
print(prediction)
model.save("model.h5")
