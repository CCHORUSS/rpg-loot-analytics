import pandas as pd
import joblib

model = joblib.load('legendary_detector.pkl')

new_item = pd.DataFrame([{
    'item_level': 90,
    'durability': 150,
    'type_accessory': 0,
    'type_armor': 0,
    'type_weapon': 1
}])

prediction = model.predict(new_item)

if prediction == 0:
    print("item ain't legendary")
elif prediction == 1:
    print("item is legendary")