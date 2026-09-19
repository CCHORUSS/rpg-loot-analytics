from random import randint

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
df = pd.DataFrame ({
    'actions_per_minute': np.random.randint(50, 351, 1000),
    'headshot_rate': np.random.randint(5, 96, 1000)})
df['is_cheater'] = ((df['actions_per_minute'] > 280) & (df['headshot_rate'] > 70)).astype(int)
print(df['is_cheater'])

X = df.drop('is_cheater', axis=1)
y  = df['is_cheater']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




model2 = LogisticRegression(class_weight='balanced')
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model2.fit(X_train_scaled, y_train)
predictions2 = model2.predict(X_test_scaled)

cm = confusion_matrix(y_test, predictions)
cm2 = confusion_matrix(y_test, predictions2)
print(cm)
print(cm2)