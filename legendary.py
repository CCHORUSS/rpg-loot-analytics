from sqlite3.dbapi2 import Date
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
n_value = 100

df = pd.DataFrame({
    'item_level': np.random.randint(1, 101, n_value),
    'durability': np.random.randint(50, 200, n_value),
    'type': np.random.choice(['weapon', 'armor', 'accessory'], n_value)

})
df['is_legendary'] = np.where((df['item_level'] > 75) & (df['type'] == 'weapon'), 1, 0)


df = pd.get_dummies(df, columns=['type'])
X = df.drop('is_legendary', axis=1)
y = df['is_legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

predictions = knn.predict(X_test_scaled)щ







RFC = RandomForestClassifier(random_state=42)
RFC.fit(X_train, y_train)
prediction = RFC.predict(X_test)




print(f"RF prediction: \n{prediction}")
print(f" Precision: {precision_score(y_test, prediction):.2f}")
print(f"F1 score: {f1_score(y_test, prediction):.2f}")
cm = confusion_matrix(y_test, prediction)
print(cm)
joblib.dump(RFC, 'legendary_detector.pkl')

print(predictions)