import pandas as pd
import numpy as np


df_heroes =  {
    "hero": ["warrior", "mage", "rouge"],
    "hp": [450, 180, 260],
    "mana": [50, 300, 120],
    "speed": [1.1, 1.0, 1.35],
    "armor_type": ["heavy", "cloth", "leather"],
    "role": ["tank", "dps", "dps"]
}
df_heroes = pd.DataFrame(df_heroes)


df_heroes["effective_hp"] = df_heroes["hp"] * df_heroes["speed"]
df_heroes["total_power"] = (df_heroes["mana"] * 1.5) + df_heroes["hp"]
#print(df_heroes[["hero", "total_power"]])
#print(df_heroes.iloc[1, :])

tanky_heroes = df_heroes[df_heroes["hp"] > 200]
#print(tanky_heroes)
fast_thanks = df_heroes[(df_heroes["hp"] > 200) & (df_heroes["speed"] >= 1.2)]
#print(fast_thanks)
#print(df_heroes.sort_values(by="hp"))
#print(df_heroes[(df_heroes["mana"] > 100) & (df_heroes["effective_hp"] >= 300)])
sorted_heroes = df_heroes.sort_values(by="effective_hp", ascending=False)
#print(sorted_heroes)
avg_hp_by_armor = df_heroes.groupby("armor_type")["hp"].mean()
#print(avg_hp_by_armor)
sorted_role = df_heroes.groupby("role")["total_power"].mean()
print(sorted_role)