import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
n_samples = 100
df = pd.DataFrame({
    'actions_per_min': np.random.randint(60, 451, n_samples),
    'mouse_entropy': np.random.randint(0.1, 1.0, n_samples),
    'client_platform': np.random.choice(['Steam', 'Epic', 'Custom_launcher'], n_samples)
})

base_condition = (df['actions_per_min'] > 360) & (df['mouse_entropy'] < 0.28)
noise = np.random.rand(len(df)) < 0.015
y = np.where(noise, 1 - base_condition.astype(int), base_condition.astype(int))
df['is_bot'] = y
df = pd.get_dummies(df, columns=['client_platform'])
X = df.drop('is_bot', axis=1)
y = df['is_bot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sclaer = StandardScaler()
X_train = sclaer.fit_transform(X_train)
X_test = sclaer.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
lr = LogisticRegression(class_weight='balanced', random_state=42)
knn.fit(X_train, y_train)
lr.fit(X_train, y_train)
predict_knn = knn.predict(X_test)
predict_lr = lr.predict(X_test)
print(predict_knn)
print(predict_lr)
f1_1 = f1_score(y_test, predict_knn)
f1_2 = f1_score(y_test, predict_lr)
cm1 = confusion_matrix(y_test, predict_knn)
cm2 = confusion_matrix(y_test, predict_lr)
print(cm1)
print(cm2)
print(f1_1)
print(f1_2)
