import json
import numpy as np
import pandas as pd


class EconomyEngine:
    def __init__(self, config_path="config.json"):
        # 1. Загружаем сырые данные из конфигурационного файла
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. Переводим списки в массивы NumPy
        self.region_taxes = np.array(data["region_taxes"])
        sword_req = data["craft_recipes"]["sword"]
        staff_req = data["craft_recipes"]["staff"]
        self.recipes = np.array([sword_req, staff_req])

        # 3. Генерируем случайные цены на складах (4 региона x 3 ресурса)
        self.stock_prices = np.random.randint(8, 25, size=(4, 3))

        # 4. Считаем цены на ресурсы с налогами через Broadcasting NumPy
        self.final_resource_prices = self.stock_prices * self.region_taxes.reshape(4, 1)

        # 5. Имена для красивых таблиц в Pandas
        self.regions = ["North", "South", "East", "West"]
        self.items = ["Sword", "Staff"]

        self.craft_prices = None

    def calculate_craft_prices(self):
        # Быстрый матричный расчёт под капотом через NumPy (@)
        raw_prices = self.final_resource_prices @ self.recipes.T

        # Упаковываем результат в Pandas DataFrame с понятными метками
        self.craft_prices = pd.DataFrame(
            raw_prices,
            index=self.regions,
            columns=self.items
        )
        return self.craft_prices

    def get_best_region_for(self, item_name):
        # Поиск лучшей цены напрямую через Pandas (.idxmin)
        best_region = self.craft_prices[item_name].idxmin()
        min_price = self.craft_prices.loc[best_region, item_name]
        return best_region, min_price

    def filter_affordable(self, item_name, max_budget):
        mask = self.craft_prices[item_name] <= max_budget
        filtered_df = self.craft_prices[mask]
        print(filtered_df)
        return filtered_df

nw = EconomyEngine("config.json")
nw.calculate_craft_prices()  # Сначала рассчитываем таблицу цен!
nw.filter_affordable("Staff", 150)


















# --- ПРОВЕРКА РАБОТЫ ДВИЖКА ---
if __name__ == "__main__":
    # Создаём экземпляр движка
    engine = EconomyEngine("config.json")

    # Считаем стоимость крафта (возвращает готовый DataFrame)
    df_prices = engine.calculate_craft_prices()
    print("=== Таблица стоимости крафта по регионам ===")
    print(df_prices)

    print("\n=== Поиск выгоды ===")
    best_region_sword, sword_price = engine.get_best_region_for("Sword")
    print(f"Лучший регион для создания Меча: {best_region_sword} (цена: {sword_price:.2f})")

    best_region_staff, staff_price = engine.get_best_region_for("Staff")
    print(f"Лучший регион для создания Посоха: {best_region_staff} (цена: {staff_price:.2f})")