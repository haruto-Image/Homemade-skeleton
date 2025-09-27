from pathlib import Path
from moviepy.editor import VideoFileClip

input_path  = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\input\0817[三野さん動画]\IMG_0085.MOV"
output_path = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\output\0817[三野さん動画]\Mino_768_768_24.mp4"

# 出力フォルダを作成（なければ作る）
out_path = Path(output_path)
out_path.parent.mkdir(parents=True, exist_ok=True)

# トリミング区間（秒）
start_time = 47
end_time   = 57

clip = VideoFileClip(input_path).subclip(start_time, end_time)

TW, TH = 768,768
W, H = clip.size

base = max(TW/W, TH/H)   # 目標のサイズを確保するために元画像を拡大縮小するときの倍率
zoom = 2.1             # 追加ズーム（≥1.0 推奨）
s = base * zoom

resized = clip.resize((int(W*s), int(H*s)))

# 中央で 800x1200 にトリム
Rw, Rh = resized.w, resized.h
x1 = max(0, (Rw - TW) // 2)
y1 = (Rh - TH) // 2 -50
out = resized.crop(x1=x1, y1=y1, x2=x1 + TW, y2=y1 + TH)



# TW = 800
# TH = 1200
# W, H = clip.w, clip.h

# # 2) 中央 1200x800 にクロップ
# x1 = (W//2) - 400
# y1 = (H//2) - 600
# out = clip.crop(x1=x1, y1=y1, x2=x1 + TW, y2=y1 + TH)

out.write_videofile(
    str(out_path), 
    codec="libx264", 
    audio_codec="aac", 
    preset="slow", 
    bitrate=None, 
    fps=24
)

