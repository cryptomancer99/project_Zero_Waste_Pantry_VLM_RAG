import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import pandas as pd
from ml_engine import PantryMLEngine
from rag_engine import RecipeRAGEngine

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("green")

class PantryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Zero-Waste Indian Pantry Planner")
        self.geometry("1100x750")

        # Initialize both backend engines
        self.ml_engine = PantryMLEngine()
        self.rag_engine = RecipeRAGEngine()
        self.selected_folder = None

        # Configure 2-column Grid layout on the window root
        self.grid_columnconfigure(0, weight=1) # Left Panel: Inputs & Controls
        self.grid_columnconfigure(1, weight=2) # Right Panel: Results View
        self.grid_rowconfigure(0, weight=1)

        # ----------------- 1. LEFT CONTROL PANEL -----------------
        # This frame belongs directly on 'self' (the window)
        self.left_panel = ctk.CTkFrame(self, corner_radius=15)
        self.left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.lbl_title = ctk.CTkLabel(self.left_panel, text="🍳 Smart Pantry AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.pack(padx=20, pady=20)

        # Folder Picker Button
        self.btn_browse = ctk.CTkButton(self.left_panel, text="📁 Select Pantry Images Folder", command=self.browse_folder)
        self.btn_browse.pack(padx=20, pady=10, fill="x")

        self.lbl_folder_status = ctk.CTkLabel(self.left_panel, text="No directory loaded.", text_color="gray")
        self.lbl_folder_status.pack(padx=20, pady=5)

        # ----------------- 2. PERSONALIZATION SUB-FRAME -----------------
        # Crucial fix: Pass self.left_panel explicitly as the master container
        self.pref_frame = ctk.CTkFrame(self.left_panel)
        self.pref_frame.pack(padx=20, pady=15, fill="x")

        self.lbl_pref_title = ctk.CTkLabel(self.pref_frame, text="⚙️ Personal Preferences", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_pref_title.pack(padx=15, pady=(10, 5), anchor="w")

        # Cuisine Selector Dropdown inside pref_frame
        self.lbl_cuisine = ctk.CTkLabel(self.pref_frame, text="Select Preferred Cuisine:")
        self.lbl_cuisine.pack(padx=15, pady=(5, 2), anchor="w")
        
        self.cuisine_options = ["Any Indian", "Punjabi", "South Indian", "Maharashtrian", "Gujarati", "Bengali", "Andhra"]
        self.combo_cuisine = ctk.CTkComboBox(self.pref_frame, values=self.cuisine_options)
        self.combo_cuisine.set("Any Indian")
        self.combo_cuisine.pack(padx=15, pady=5, fill="x")

        # Allergy Entry Box inside pref_frame
        self.lbl_allergies = ctk.CTkLabel(self.pref_frame, text="Enter Vegetable Allergies (comma separated):")
        self.lbl_allergies.pack(padx=15, pady=(10, 2), anchor="w")
        
        self.entry_allergies = ctk.CTkEntry(self.pref_frame, placeholder_text="e.g., Tomato, Potato")
        self.entry_allergies.pack(padx=15, pady=5, fill="x")

        # ----------------- 3. CORE ACTION BUTTON -----------------
        self.btn_process = ctk.CTkButton(self.left_panel, text="🚀 Scan Pantry & Find Recipes", 
                                          command=self.process_pipeline, state="disabled", fg_color="#2b7de2", hover_color="#1e5cb3")
        self.btn_process.pack(padx=20, pady=25, fill="x")


        # ----------------- 4. RIGHT DISPLAY PANEL -----------------
        self.right_panel = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=2)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Top Right: Inventory Table Display
        self.txt_inventory = ctk.CTkTextbox(self.right_panel, font=ctk.CTkFont(family="Courier", size=13))
        self.txt_inventory.grid(row=0, column=0, padx=0, pady=(0, 10), sticky="nsew")
        self.txt_inventory.insert("1.0", "=== Scanned Vegetable Inventory Output ===")

        # Bottom Right: RAG Recipe Output
        self.txt_recipes = ctk.CTkTextbox(self.right_panel, font=ctk.CTkFont(size=14))
        self.txt_recipes.grid(row=1, column=0, padx=0, pady=(10, 0), sticky="nsew")
        self.txt_recipes.insert("1.0", "=== Personalized Vegetarian Recommendations ===")

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.lbl_folder_status.configure(text=f"Loaded: ...{os.path.basename(folder)}", text_color="lightgreen")
            self.btn_process.configure(state="normal")

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

        # Trigger Model Object Detection
        df_total, df_usable = self.ml_engine.process_pantry_directory(self.selected_folder)

        if df_total.empty:
            self.txt_inventory.insert("end", "\nNo vegetables found in directory images.")
            self.txt_recipes.insert("end", "\nPantry empty. Cannot fetch recipe matches.")
            return

        # Display clean text layout table to user
        self.txt_inventory.delete("1.0", "end")
        self.txt_inventory.insert("end", "=== CURRENT SCAN TABLE INVENTORY ===\n\n")
        self.txt_inventory.insert("end", df_total.to_string(index=False))

        # Trigger Recipe RAG Search with Filters
        self.txt_recipes.insert("end", "Filtering 6000+ entries for allergen-free vegetarian recipes...\n")
        self.update_idletasks()
        
        top_recipes = self.rag_engine.find_best_recipes(
            df_usable_inventory=df_usable,
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

if __name__ == "__main__":
    app = PantryApp()
    app.mainloop()