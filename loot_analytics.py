import numpy as np
import pandas as pd


class LootSystem:
    def __init__(self, num_kills):
        self.gold = np.random.randint(10, 100, size=num_kills)
        self.exp = np.random.randint(50, 200, size=num_kills)
        self.rarity_roll = np.random.rand(num_kills)

        monsters_types = ["Goblin", "Orc", "Dragon"]
        self.monsters = np.random.choice(monsters_types, size=num_kills, p=[0.6, 0.35, 0.05])

        multipliers = {"Goblin": 1.0, "Orc": 2.0, "Dragon": 10.0}


        self.df = pd.DataFrame(
            {
                "gold": self.gold,
                "exp": self.exp,
                "rarity_roll": self.rarity_roll,
                "monsters": self.monsters
            }
        )
        self.df["gold_multiplier"] = self.df["monsters"].map(multipliers)
        self.df["final_gold"] = self.df["gold"] * self.df["gold_multiplier"]
        self.df["rarity"] = self.df["rarity_roll"].map(classify)
        self.df.to_csv("farm_results.csv", index=False)


    def get_legendary_drop(self, thresh_hold=0.95):
        legendary_drop = self.df[self.df["rarity_roll"] >= thresh_hold]
        return legendary_drop

    def get_summary(self):
        print(f"total amount of collected gold {self.df['final_gold'].sum()}")
        print(f"average amount of collected exp {self.df['exp'].mean()}")
        legendary_df = self.get_legendary_drop()
        count = legendary_df.shape[0]
        print(f"total amount of collected LD {count}")
        return count, legendary_df

    def get_stats_by_monster(self):
        monster_stats = self.df.groupby("monsters").agg({"final_gold": "sum", "exp": "mean"})
        print(monster_stats)
        return monster_stats

    def save_to_csv(self, filename="farm_results.csv"):
        # index=False отключает сохранение числовых номеров строк (0, 1, 2...)
        self.df.to_csv(filename, index=False)
        print(f"Данные успешно сохранены в файл: {filename}")

    def load_from_csv(self, filename="farm_results.csv"):
        # Просто перезаписываем текущую таблицу данными из файла
        self.df = pd.read_csv(filename)

        # И сразу запускаем методы анализа
        self.get_summary()
        self.get_stats_by_monster()

def classify(roll):
    if roll >= 0.95:
        return "legendary"
    elif roll >= 0.80:
        return "rarity"
    return "common"



if __name__ == "__main__":
    loot = LootSystem(1000)
    print(loot.df["rarity"].value_counts())

    # 1. Показать первые 5 строк всей таблицы
    print("--- Первые 5 убийств ---")
    print(loot.df.head())

    print("\n--- Сводка по фарму ---")
    loot.get_summary()

    print("\n--- Легендарный дроп ---")
    legendaries = loot.get_legendary_drop(0.95)
    print(legendaries.head())

    loot.get_stats_by_monster()



