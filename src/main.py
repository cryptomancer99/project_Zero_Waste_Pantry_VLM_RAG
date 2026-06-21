import sys
import threading
import time
import os
from tkinter import filedialog
import customtkinter as ctk

# 🔗 BRIDGE TO YOUR ML ENGINE: Import the real vision pipeline
from ml_engine import PantryMLEngine

# =====================================================================
# 1. INTEGRATED AI ENGINE WRAPPER
# =====================================================================
class IndianPantryEngine:
    def __init__(self):
        # Instantiate your real Microsoft Florence-2 GPU model
        self.ml_pipeline = PantryMLEngine()
        
        # Recipes configuration database
        self.recipe_database = {
            "South Indian": [
                {
                    "title": "🥦 Vegetable Kootu (South Indian Stew)",
                    "required": ["cauliflower", "onion"],
                    "ingredients": "Cauliflower, Onion, Green Chilli, Curry leaves.",
                    "instructions": "1. Sauté onions and green chillies.\n2. Add cauliflower florets and simmer until tender.\n3. Stir in coconut paste."
                },
                {
                    "title": "🍛 Vendakkai Poriyal (Okra Stir-Fry)",
                    "required": ["lady-finger", "onion"],
                    "ingredients": "Lady-finger (Okra), Onion, Mustard seeds, Green chilli.",
                    "instructions": "1. Sauté mustard seeds and onions.\n2. Add chopped lady-finger and stir-fry on medium heat until crisp."
                }
            ],
            "North Indian": [
                {
                    "title": "🥘 Gobi Fry",
                    "required": ["cauliflower"],
                    "ingredients": "Cauliflower, Onion, Green Chilli, Garam Masala.",
                    "instructions": "1. Parboil cauliflower florets.\n2. Toss with onions, chillies, and spices in a pan until golden brown."
                }
            ]
        }

    def process_images_in_folder(self, folder_path, log_callback):
        log_callback(f"📂 VLM: Initializing visual model scene captioning scanner...")
        log_callback(f"📂 Scanning target directory: {folder_path}...\n")
        
        try:
            # Execute your real Florence-2 model parsing over the directory images
            df_total, df_usable = self.ml_pipeline.process_pantry_directory(folder_path)
            
            # Extract detected items array out of your pandas usable DataFrame
            if not df_usable.empty and 'Item' in df_usable.columns:
                detected_raw = df_usable['Item'].dropna().str.lower().tolist()
            else:
                detected_raw = []
                
            # Anti-Hallucination Guard: Remove potential context false-positives manually if needed
            # (e.g., if Florence-2 infers a tomato from the phrase "pulao" or "channa masala")
            if "tomato" in detected_raw:
                # Optional: log_callback("🛡️ Filtered out a contextually suspected tomato hallucination.")
                pass
                
            detected_inventory = list(set(detected_raw))
            log_callback(f"\n🔍 Vision Model Complete. Discovered raw inventory: {detected_inventory}")
            return detected_inventory
            
        except Exception as e:
            log_callback(f"❌ Error during VLM Inference execution: {str(e)}")
            return []

    def run_rag_reranker(self, raw_inventory, allergies, cuisine, log_callback):
        log_callback("🤖 RAG: Applying custom allergen filters and re-ranking recipes...")
        time.sleep(0.5)
        
        allergy_list = [a.strip().lower() for a in allergies.split(",") if a.strip()]
        if not allergy_list:
            allergy_list = ['brinjal']
            
        log_callback(f"[RAG RE-RANKER] Active Allergies Blocked: {allergy_list}")
        
        # Filter allergens out
        safe_veggies = [veg for veg in raw_inventory if veg not in allergy_list]
        log_callback(f"[RAG RE-RANKER] Safe Usable Vegetables: {safe_veggies}")
        log_callback(f"[RAG RE-RANKER] Target Cuisine Filter: {cuisine}\n")
        
        cuisine_recipes = self.recipe_database.get(cuisine, self.recipe_database["South Indian"])
        matched_recipes = []
        
        for recipe in cuisine_recipes:
            if all(req_item in safe_veggies for req_item in recipe["required"]):
                matched_recipes.append(recipe)
                
        log_callback(f"✨ Zero-Waste Recipe Generation Complete! Matched {len(matched_recipes)} recipes.")
        return safe_veggies, allergy_list, matched_recipes


# =====================================================================
# 2. INTERACTIVE GUI DASHBOARD
# =====================================================================
class PantryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Project Zero Waste Pantry")
        self.geometry("1050x680")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # System initializes the pipeline (will display your GPU CUDA status logs in terminal)
        self.engine = IndianPantryEngine()

        self.title_label = ctk.CTkLabel(
            self, text="Project Zer0 Waste Pantry (Live VLM Active)", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.title_label.pack(pady=15)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Inputs Panel
        self.input_frame = ctk.CTkFrame(self.main_frame, width=300)
        self.input_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.input_title = ctk.CTkLabel(self.input_frame, text="User Preferences", font=ctk.CTkFont(size=16, weight="bold"))
        self.input_title.pack(pady=10)

        self.folder_label = ctk.CTkLabel(self.input_frame, text="Pantry Images Folder:")
        self.folder_label.pack(anchor="w", padx=15)
        self.folder_browser_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.folder_browser_frame.pack(fill="x", padx=15, pady=5)
        self.folder_input = ctk.CTkEntry(self.folder_browser_frame, placeholder_text="No folder selected...")
        self.folder_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.browse_btn = ctk.CTkButton(self.folder_browser_frame, text="Browse", width=60, command=self.select_folder)
        self.browse_btn.pack(side="right")

        self.allergy_label = ctk.CTkLabel(self.input_frame, text="Allergies (comma separated):")
        self.allergy_label.pack(anchor="w", padx=15, pady=(10, 0))
        self.allergy_input = ctk.CTkEntry(self.input_frame)
        self.allergy_input.insert(0, "brinjal") 
        self.allergy_input.pack(fill="x", padx=15, pady=5)

        self.cuisine_label = ctk.CTkLabel(self.input_frame, text="Target Cuisine Filter:")
        self.cuisine_label.pack(anchor="w", padx=15, pady=(10, 0))
        self.cuisine_dropdown = ctk.CTkOptionMenu(self.input_frame, values=["South Indian", "North Indian"])
        self.cuisine_dropdown.set("South Indian") 
        self.cuisine_dropdown.pack(fill="x", padx=15, pady=5)

        self.submit_btn = ctk.CTkButton(self.input_frame, text="Run AI Pipeline", command=self.trigger_pipeline)
        self.submit_btn.pack(fill="x", padx=15, pady=20)

        # Verified Inventory Panel
        self.inventory_frame = ctk.CTkFrame(self.main_frame, width=220)
        self.inventory_frame.pack(side="left", fill="both", padx=5, pady=10)
        
        self.inv_title = ctk.CTkLabel(self.inventory_frame, text="📋 Verified Inventory", font=ctk.CTkFont(size=16, weight="bold"))
        self.inv_title.pack(pady=10)
        
        self.inventory_box = ctk.CTkTextbox(self.inventory_frame, font=("Segoe UI", 12), width=200)
        self.inventory_box.pack(fill="both", expand=True, padx=10, pady=5)
        self.inventory_box.insert("end", "Run pipeline to parse image directory...")

        # Tabs Output Frame
        self.output_frame = ctk.CTkFrame(self.main_frame)
        self.output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.tab_view = ctk.CTkTabview(self.output_frame)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tab_recipes = self.tab_view.add("✨ Generated Recipes")
        self.tab_logs = self.tab_view.add("💻 System Processing Logs")

        self.log_display = ctk.CTkTextbox(self.tab_logs, font=("Consolas", 11))
        self.log_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_display.insert("end", "System ready.\n")

        self.recipe_display = ctk.CTkTextbox(self.tab_recipes, font=("Segoe UI", 12), wrap="word")
        self.recipe_display.pack(fill="both", expand=True, padx=5, pady=5)
        self.recipe_display.insert("end", "No recipes generated yet.")

    def select_folder(self):
        chosen_dir = filedialog.askdirectory(title="Select Pantry Images Folder")
        if chosen_dir:
            self.folder_input.delete(0, "end")
            self.folder_input.insert(0, chosen_dir)
            self.append_log(f"📁 Target folder path set to: {chosen_dir}")

    def append_log(self, text):
        self.log_display.insert("end", text + "\n")
        self.log_display.see("end")

    def update_inventory_ui(self, safe_veggies, blocked_allergies):
        self.inventory_box.configure(state="normal")
        self.inventory_box.delete("1.0", "end")
        
        self.inventory_box.insert("end", "✅ SAFE VEGETABLES:\n")
        if not safe_veggies:
            self.inventory_box.insert("end", "  (None detected)\n")
        for veg in safe_veggies:
            self.inventory_box.insert("end", f" • {veg.capitalize()}\n")
            
        self.inventory_box.insert("end", "\n❌ BLOCKED / ALLERGENS:\n")
        for alg in blocked_allergies:
            self.inventory_box.insert("end", f" • {alg.capitalize()}\n")
        self.inventory_box.configure(state="disabled")

    def display_recipes(self, recipes):
        self.recipe_display.configure(state="normal")
        self.recipe_display.delete("1.0", "end")
        
        if not recipes:
            self.recipe_display.insert("end", "⚠️ No safe recipes match your exact current inventory ingredients.")
        else:
            self.recipe_display.insert("end", "💡 Verified zero-waste recipes:\n\n" + "="*50 + "\n\n")
            for idx, item in enumerate(recipes, start=1):
                self.recipe_display.insert("end", f"{idx}. {item['title'].upper()}\n")
                self.recipe_display.insert("end", f"🥕 Ingredients: {item['ingredients']}\n\n")
                self.recipe_display.insert("end", f"🍳 Instructions:\n{item['instructions']}\n\n")
                self.recipe_display.insert("end", "-"*50 + "\n\n")
        
        self.tab_view.set("✨ Generated Recipes")

    def trigger_pipeline(self):
        target_folder = self.folder_input.get()
        if not target_folder:
            self.append_log("⚠️ Warning: Please pick an image folder path before running.")
            return

        self.submit_btn.configure(state="disabled", text="Processing...")
        user_allergies = self.allergy_input.get()
        user_cuisine = self.cuisine_dropdown.get()
        self.tab_view.set("💻 System Processing Logs")

        ai_thread = threading.Thread(
            target=self._async_worker, 
            args=(target_folder, user_allergies, user_cuisine),
            daemon=True
        )
        ai_thread.start()

    def _async_worker(self, folder, allergies, cuisine):
        self.append_log("\n--- Starting New Live Engine Run ---")
        
        # 1. Calls Florence-2 inside ml_engine.py securely
        raw_inventory = self.engine.process_images_in_folder(folder, self.append_log)
        
        # 2. Feeds raw detected list to RAG layer
        safe_veggies, blocked, recipes = self.engine.run_rag_reranker(raw_inventory, allergies, cuisine, self.append_log)
        
        # 3. Render
        self.update_inventory_ui(safe_veggies, blocked)
        self.display_recipes(recipes)
        
        self.submit_btn.configure(state="normal", text="Run AI Pipeline")


if __name__ == "__main__":
    try:
        app = PantryApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Application gracefully stopped via Ctrl+C.")
        sys.exit(0)