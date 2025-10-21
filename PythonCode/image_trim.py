from pathlib import Path
from PIL import Image, ImageOps

input_path  = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\output\1003\UedaB_W.png"
output_path = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\output\1003\UedaB_W2.png"

TW, TH = 800,1200
img = ImageOps.exif_transpose(Image.open(input_path))  # EXIFの回転を画素に適用してから使う
W, H = img.size

base = max(TW/W, TH/H)   # 目標のサイズを確保するために元画像を拡大縮小するときの倍率
zoom = 1    # 追加ズーム（≥1.0 推奨）
s = base * zoom

resized = img.resize((int(W*s), int(H*s)), Image.LANCZOS)

# 中央で 800x1200 にトリム
x1 = (resized.width  - TW) // 2 +10
y1 = (resized.height -TH)//2 
out = resized.crop((x1, y1, x1 + TW, y1 + TH))
out.save(output_path, quality=95)
