import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch

# Workaround for Hugging Face's strict flash_attn regex scanner on Windows
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

class PantryMLEngine:
    def __init__(self):
        print("\n=== Initializing High-Speed GPU Indian Pantry Engine ===")
        
        # Verify and lock onto NVIDIA CUDA cores
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model_id = "microsoft/Florence-2-base"
        
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=self.torch_dtype,
                attn_implementation="eager"
            ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print(f"🚀 SUCCESS: Model anchored directly to: {self.device.upper()}!")

        # Dictionary of structural variants to parse out of descriptive strings
        self.search_dictionary = {
            "tomato": "Tomato", "tomatoes": "Tomato",
            "onion": "Onion", "onions": "Onion",
            "potato": "Potato", "potatoes": "Potato",
            "cauliflower": "Cauliflower", "gobi": "Cauliflower",
            "radish": "Radish", "white radish": "White Radish", "daikon": "White Radish", "mooli": "White Radish",
            "lady-finger": "Lady-finger", "lady finger": "Lady-finger", "okra": "Lady-finger", "bhindi": "Lady-finger",
            "green chilli": "Green Chilli", "green chili": "Green Chilli", "chilli": "Green Chilli", "chili": "Green Chilli", "pepper": "Green Chilli",
            "coriander": "Coriander", "cilantro": "Coriander", "dhania": "Coriander",
            "palak": "Palak", "spinach": "Palak",
            "pumpkin": "Pumpkin", "kaddu": "Pumpkin",
            "brinjal": "Brinjal", "eggplant": "Brinjal", "aubergine": "Brinjal", "baingan": "Brinjal",
            "carrot": "Carrot", "carrots": "Carrot"
        }

    def process_pantry_directory(self, directory_path):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")

        supported_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported_extensions)]

        all_detected_items = []

        for img_file in image_files:
            img_path = os.path.join(directory_path, img_file)
            print(f"Processing {img_file}...", end="", flush=True)
            try:
                raw_image = Image.open(img_path).convert("RGB")
                
                # Use detailed text scene description to avoid dataset label limitations
                scene_description = self._run_detailed_caption(raw_image).lower()
                
                match_count = 0
                found_in_this_image = set()

                # Search the text description for known ingredients
                for keyword, standardized_name in self.search_dictionary.items():
                    if keyword in scene_description and standardized_name not in found_in_this_image:
                        # Baseline default freshness assessment
                        all_detected_items.append({'Item': standardized_name, 'Status': 'Fresh'})
                        found_in_this_image.add(standardized_name)
                        match_count += 1
                
                print(f" Done! Found {match_count} verified vegetables.")
                        
            except Exception as e:
                print(f" ❌ Error processing image: {e}")

        if not all_detected_items:
            return pd.DataFrame(columns=['Item', 'Count']), pd.DataFrame(columns=['Item', 'Count'])

        df_raw = pd.DataFrame(all_detected_items)
        df_total = df_raw.groupby('Item').size().reset_index(name='Count')
        df_usable = df_raw[df_raw['Status'] == 'Fresh'].groupby('Item').size().reset_index(name='Count')
        
        return df_total, df_usable

    def _run_detailed_caption(self, pil_image):
        """Runs an ensemble strategy combining two distinct descriptive viewpoints."""
        combined_descriptions = ""
        
        # We loop through two different prompt tasks to maximize what the CPU model extracts
        for prompt in ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
            try:
                inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device, self.torch_dtype)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        num_beams=3,
                        early_stopping=True
                    )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=prompt, 
                    image_size=pil_image.size
                )
                combined_descriptions += " " + parsed_answer.get(prompt, "")
            except Exception:
                continue

        return combined_descriptions