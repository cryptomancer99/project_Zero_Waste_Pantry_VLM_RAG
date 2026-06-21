def process_pipeline(self):
        self.txt_inventory.delete("1.0", "end")
        self.txt_recipes.delete("1.0", "end")
        
        self.txt_inventory.insert("end", "Parsing images via Visual Language Model CPU Core...\n")
        self.update_idletasks()

        # Gather Preferences from user inputs
        cuisine_pref = self.combo_cuisine.get()
        if cuisine_pref == "Any Indian":
            cuisine_pref = None
            
        raw_allergies = self.entry_allergies.get()
        allergy_list = [a.strip() for a in raw_allergies.split(",") if a.strip()]
        allergy_list_lower = [a.lower() for a in allergy_list]

        # Trigger Model Object Detection
        df_total, df_usable = self.ml_engine.process_pantry_directory(self.selected_folder)

        if df_total.empty:
            self.txt_inventory.insert("end", "\nNo vegetables found in directory images.")
            self.txt_recipes.insert("end", "\nPantry empty. Cannot fetch recipe matches.")
            return

        # ─── CORE FIX: FILTER USABLE INVENTORY AGAINST ALLERGIES ───
        if not df_usable.empty:
            df_safe_usable = df_usable[~df_usable['Item'].str.lower().isin(allergy_list_lower)]
        else:
            df_safe_usable = df_usable.copy()

        # Render both tables cleanly to the UI pane
        self.txt_inventory.delete("1.0", "end")
        self.txt_inventory.insert("end", "=== ALL SCANNED ITEMS (TOTAL) ===\n")
        self.txt_inventory.insert("end", df_total.to_string(index=False))
        
        self.txt_inventory.insert("end", "\n\n✅ FRESH & ALLERGEN-SAFE INVENTORY ===\n")
        if df_safe_usable.empty:
            self.txt_inventory.insert("end", "(No safe, fresh items available)")
        else:
            self.txt_inventory.insert("end", df_safe_usable.to_string(index=False))

        # Trigger Recipe RAG Search with Filters
        self.txt_recipes.insert("end", "Filtering 6000+ entries for allergen-free vegetarian recipes...\n")
        self.update_idletasks()
        
        top_recipes = self.rag_engine.find_best_recipes(
            df_usable_inventory=df_safe_usable,  # Send the pre-filtered safe inventory to RAG
            allergies=allergy_list,
            preferred_cuisine=cuisine_pref,
            top_n=3
        )

        # Render clean results
        self.txt_recipes.delete("1.0", "end")
        if not top_recipes:
            self.txt_recipes.insert("end", f"⚠️ No vegetarian recipes found matching your safe pantry items for cuisine: {cuisine_pref or 'All'}")
            return

        self.txt_recipes.insert("end", f"🍳 TOP {len(top_recipes)} PERSONALIZED VEGETARIAN DISHES DISCOVERED 🍳\n\n")
        for i, recipe in enumerate(top_recipes, 1):
            self.txt_recipes.insert("end", f"👉 {i}. {recipe['recipe_name']} [{recipe['cuisine']} Cuisine]\n")
            self.txt_recipes.insert("end", f"   ✅ Safe Items Used: {', '.join(recipe['matched'])}\n")
            if recipe['missing']:
                self.txt_recipes.insert("end", f"   ⚠️ Missing Ingredients: {', '.join(recipe['missing'])}\n")
            self.txt_recipes.insert("end", f"   📖 Recipe Instructions:\n   {recipe['instructions']}\n\n")
            self.txt_recipes.insert("end", "-"*80 + "\n\n")