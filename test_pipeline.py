import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
np.random.seed(42)
n_samples = 500


df = pd.DataFrame({
    'session_hours': np.random.randint(1, 25, n_samples),
    'actions_per_min': np.random.randint(10, 401, n_samples),
    'server_region': np.random.choice(['EU', 'NA', 'ASIA'], n_samples)
})
df['is_bot'] = np.where((df['session_hours'] > 18) & (df['actions_per_min'] > 300), 1, 0)
df = pd.get_dummies(df, columns=['server_region'])
X = df.drop('is_bot', axis=1)
y = df['is_bot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
prediction = model.predict(X_test_scaled)
print(prediction)
cm = confusion_matrix(y_test, prediction)
fscore = f1_score(y_test, prediction)
print(cm)
print(fscore)