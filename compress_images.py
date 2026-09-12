
import os
import glob
from PIL import Image

image_dir = r"C:\Users\doks\Desktop\cooking_my_thesis\export_latex\images"
pngs = glob.glob(os.path.join(image_dir, "*.png"))

for png in pngs:
    img = Image.open(png)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize if too large to save space
    if img.width > 1200:
        ratio = 1200.0 / img.width
        new_size = (1200, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
    jpg_path = png.replace(".png", ".jpg")
    img.save(jpg_path, "JPEG", quality=75, optimize=True)
    os.remove(png)

# Update tex files
tex_dir = r"C:\Users\doks\Desktop\cooking_my_thesis\export_latex\chapters"
tex_files = glob.glob(os.path.join(tex_dir, "*.tex"))
for tex in tex_files:
    with open(tex, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(".png}", ".jpg}")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(content)

print("Compression complete!")

