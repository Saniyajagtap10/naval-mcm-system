import os
import glob
from PIL import Image
import streamlit as st  # <-- He sabse varti ahe ka bagh!

# Page config
st.set_page_config(
    page_title="PRJ44 Sonar Engine", page_icon="⚓", layout="wide"
)

st.title("PRJ44_Sonar_Engine (Navy HUD)")
st.write(
    "Analyze individual acoustic files with AI and real-time dynamic frequency"
    " spectrum analysis."
)

# 1. Dataset madhun sagle photos recursive (subfolders suddha) shodhnyasathi:
dataset_dir = "dataset"
image_files = []

if os.path.exists(dataset_dir):
  for root, dirs, files in os.walk(dataset_dir):
    for file in files:
      if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_files.append(os.path.join(root, file))

# 2. Dropdown madhe dakhavnyasathi
if not image_files:
  st.error(
      "❌ कृपया 'dataset' फोल्डरमध्ये किंवा त्याच्या सबफोल्डरमध्ये काही"
      " इमेजेस टाका!"
  )
else:
  selected_file = st.selectbox(
      "Select Target File from Dataset:", image_files
  )

  if selected_file:
    try:
      # Image safely open karnyasathi
      raw_pil = Image.open(selected_file).convert("RGB")
      st.image(raw_pil, caption=f"Selected: {selected_file}", use_column_width=True)
      st.success("Target loaded successfully for AI Acoustic Scan!")
    except Exception as e:
      st.error(f"Error loading image: {e}")
