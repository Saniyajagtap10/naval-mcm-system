import glob
import os
from PIL import Image

# 1. Dataset madhun sagle photos recursive (subfolders suddha) shodhnyasathi:
dataset_dir = "dataset"
image_files = []

if os.path.exists(dataset_dir):
  # Sagle subfolders madle jpg/png photos shodhail
  for root, dirs, files in os.walk(dataset_dir):
    for file in files:
      if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_files.append(os.path.join(root, file))

# 2. Dropdown madhe dakhavnyasathi
if not image_files:
  st.error("❌ कृपया 'dataset' फोल्डरमध्ये किंवा त्याच्या सबफोल्डरमध्ये काही इमेजेस टाका!")
else:
  selected_file = st.selectbox(
      "Select Target File from Dataset:", image_files
  )

  if selected_file:
    try:
      # Image safely open karnyasathi
      raw_pil = Image.open(selected_file).convert("RGB")
      # Baki tumcha AI scan code ithe yeyil...
    except Exception as e:
      st.error(f"Error loading image: {e}")
