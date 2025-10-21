import cv2  # OpenCVライブラリを追加
from pathlib import Path
from PIL import Image

# --- 設定項目 ---

# 1. 入力する動画ファイルのパス
input_video_path = Path(r"C:\Users\_s2520798\Documents\1.研究\入出力映像\input\1003\植田部長.MOV")

# 2. 画像として保存する際の出力パス
output_image_path = Path(r"C:\Users\_s2520798\Documents\1.研究\入出力映像\output\1003\UedaB.jpg")

# 3. 切り出したいフレームの番号 (例: 300フレーム目)
TARGET_FRAME_NUMBER = 4680

# 4. 出力画像の目標サイズ (幅 と 高さ)
TW, TH = 800, 1200

# 5. ズーム倍率 (1.0以上を推奨)
zoom = 2.3

# --- ここから処理 ---

# 1. 動画ファイルから指定したフレームを読み込む
cap = cv2.VideoCapture(str(input_video_path))

if not cap.isOpened():
    print(f"エラー: 動画ファイルが開けませんでした: {input_video_path}")
else:
    # 総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画の総フレーム数: {total_frames}")

    if TARGET_FRAME_NUMBER >= total_frames:
        print(f"エラー: 指定されたフレーム番号 ({TARGET_FRAME_NUMBER}) が総フレーム数を超えています。")
    else:
        # 指定したフレーム番号に移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME_NUMBER)#cv2.CAP_PROP_POS_FRAMESはsetで変更したいプロパティの種類を指定している
        
        # フレームを1枚読み込む
        success, frame = cap.read()#Pythonがタプルという１つのオブジェクトにまとめる,cap.readは１番目の要素をbool,2つ目を画像データと設定している
        
        if success:
            # OpenCVで読み込んだ画像はBGR形式なので、Pillowで扱うためにRGB形式に変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # OpenCVの画像データ(numpy配列)をPillowのImageオブジェクトに変換
            img = Image.fromarray(frame_rgb)

            # 2. 読み込んだフレーム（画像）をリサイズ・中央トリミング (ここからは元のコードとほぼ同じ)
            W, H = img.size

            base = max(TW / W, TH / H)  # 目標のサイズを確保するために元画像を拡大縮小するときの倍率
            s = base * zoom

            resized = img.resize((int(W * s), int(H * s)), Image.LANCZOS)

            # 中央で 800x1200 にトリム
            x1 = (resized.width - TW) // 2 -100
            # y1の "+ 20" は、トリミング位置を少し下にずらす処理です（元のコードから維持）
            y1 = (resized.height - TH) // 2 - 50
            out = resized.crop((x1, y1, x1 + TW, y1 + TH))
            
            # 結果を保存
            out.save(output_image_path, quality=95)
            print(f"動画 '{input_video_path.name}' の {TARGET_FRAME_NUMBER} フレーム目を切り出し、リサイズして '{output_image_path.name}' に保存しました。")
        else:
            print(f"エラー: {TARGET_FRAME_NUMBER} フレーム目の読み込みに失敗しました。")

    # 動画ファイルを閉じる
    cap.release()
