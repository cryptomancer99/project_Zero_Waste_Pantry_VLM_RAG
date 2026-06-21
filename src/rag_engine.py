import os
import urllib.request
import pandas as pd

class RecipeRAGEngine:
    def __init__(self, db_path="data/cleaned_indian_recipes.csv"):
        self.db_path = db_path
        self.remote_url = "https://raw.githubusercontent.com/Sachinart/Indian-Recipe-API/master/IndianFoodDataset.csv"
        self.recipes_df = self._initialize_database()

    def _initialize_database(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.db_path):
            print("\n📥 Local recipe database empty. Fetching thousands of Indian recipes...")
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(self.remote_url, self.db_path)
                print("✅ Success! 6000+ Indian Recipes cached locally.")
            except Exception as e:
                print(f"❌ Failed to fetch dataset: {e}.")
                return pd.DataFrame()
        try:
            return pd.read_csv(self.db_path, encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f"❌ Error initializing dataset: {e}")
            return pd.DataFrame()

    def _clean_text(self, cell_value):
        text = str(cell_value).strip()
        if "\n" in text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            english_lines = [l for l in lines if not any(ord(c) > 2300 and ord(c) < 2432 for c in l)]
            if english_lines:
                return " ".join(english_lines)
        return text

    def find_best_recipes(self, df_usable_inventory, allergies=None, preferred_cuisine=None, top_n=3, min_match_percentage=0.30):
        """
        Filters out allergen ingredients, screens by preferred regional cuisines, 
        and extracts strict vegetarian English recipe matches.
        """
        if self.recipes_df.empty or df_usable_inventory.empty:
            print("⚠️ Engine processing halted: Dataset or scan table matrix is empty.")
            return []

        # 1. Clean and apply allergy restrictions to the inventory dataframe
        allergy_list = [a.lower().strip() for a in (allergies or [])]
        df_filtered_inventory = df_usable_inventory[~df_usable_inventory['Item'].str.lower().str.strip().isin(allergy_list)]
        
        available_veggies = [v.lower().strip() for v in df_filtered_inventory['Item'].tolist()]
        
        print(f"\n[RAG RE-RANKER] Active Allergies Blocked: {allergy_list if allergy_list else 'None'}")
        print(f"[RAG RE-RANKER] Safe Usable Vegetables: {available_veggies}")
        if preferred_cuisine:
            print(f"[RAG RE-RANKER] Target Cuisine Filter: {preferred_cuisine}")

        scored_recipes = []
        all_tracked_vegetables = ["potato", "onion", "tomato", "cauliflower", "radish", "white radish", "lady-finger", "okra", "green chilli", "green chili", "coriander", "palak", "spinach"]
        non_veg_keywords = ["chicken", "mutton", "fish", "prawn", "egg ", "eggcurry", "shrimp", "lamb", "pork", "beef"]

        for _, row in self.recipes_df.iterrows():
            raw_name = row.get('TranslatedRecipeName', row.get('RecipeName', 'Unknown Dish'))
            raw_ingredients = row.get('TranslatedIngredients', row.get('Ingredients', ''))
            raw_instructions = row.get('TranslatedInstructions', row.get('Instructions', ''))
            cuisine_tag = str(row.get('Cuisine', '')).strip()

            recipe_name = self._clean_text(raw_name).replace("RecipeName", "").replace("TranslatedRecipeName", "").strip()
            ingredients_text = str(raw_ingredients).lower()
            instructions = self._clean_text(raw_instructions).replace("Instructions", "").replace("TranslatedInstructions", "").strip()

            # --- Strict Vegetarian Filter ---
            if any(kw in recipe_name.lower() or kw in ingredients_text for kw in non_veg_keywords):
                continue

            # --- Strict Allergy Safety Filter ---
            # Even if the ingredient isn't on our tracked list, if it's explicitly written in the recipe text, dump it!
            if any(allergen in ingredients_text or allergen in recipe_name.lower() for allergen in allergy_list):
                continue

            # --- Cuisine Match Filter ---
            if preferred_cuisine and preferred_cuisine.lower() not in cuisine_tag.lower():
                continue

            # Calculate match metrics
            recipe_needed_veggies = [v for v in all_tracked_vegetables if v in ingredients_text]
            if not recipe_needed_veggies:
                continue

            matched_veggies = [v for v in recipe_needed_veggies if v in available_veggies]
            match_percentage = len(matched_veggies) / len(recipe_needed_veggies)

            if match_percentage >= min_match_percentage:
                missing_veggies = [v for v in recipe_needed_veggies if v not in available_veggies]
                
                scored_recipes.append({
                    "recipe_name": recipe_name,
                    "cuisine": cuisine_tag,
                    "match_score_pct": match_percentage,
                    "matched": [m.capitalize() for m in matched_veggies],
                    "missing": [ms.capitalize() for ms in missing_veggies],
                    "instructions": instructions[:400] + "..." if len(instructions) > 400 else instructions
                })

        # Sort based on highest ingredient matches first
        scored_recipes = sorted(scored_recipes, key=lambda x: (x["match_score_pct"], -len(x["missing"])), reverse=True)
        return scored_recipes[:top_n]