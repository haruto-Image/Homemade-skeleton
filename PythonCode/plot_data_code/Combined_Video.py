import cv2
import numpy as np
import os

def combine_videos_with_padding(video_path1, video_path2, output_path):
    """
    解像度が異なる2つの動画を、リサイズせずに横に結合する。
    高さが足りない方には黒帯（パディング）が追加される。
    """
    
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened():
        print(f"エラー: {video_path1} を開けません。")
        return
    if not cap2.isOpened():
        print(f"エラー: {video_path2} を開けません。")
        cap1.release()
        return

    # --- 動画の仕様を取得 ---
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # video1 のサイズ
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # video2 のサイズ
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- ★ ここが重要な変更点 ---
    # 最終的な出力動画のサイズを計算
    # 幅 = 2つの動画の幅の合計
    output_width = width1 + width2
    # 高さ = 2つの動画のうち、「高い方」の高さ
    output_height = max(height1, height2)
    # --- ★ 変更点ここまで ---

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"保存先フォルダ {output_dir} が存在しないため作成します。")
        os.makedirs(output_dir)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        print(f"エラー: {output_path} への書き込みを開始できません。")
        cap1.release()
        cap2.release()
        return

    print(f"動画の結合（パディングあり）を開始します... (保存先: {output_path})")
    print(f"出力解像度: {output_width} x {output_height}")

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # フレームが読み込めなかった場合（終端または破損）
        if not ret1 or not ret2 or frame1 is None or frame2 is None:
            print(f"フレーム {frame_count} で動画の終端または破損フレームに到達したため、処理を終了します。")
            break

        # --- ★ ここが重要な変更点 ---
        # 1. 黒い背景（キャンバス）を作成
        #    np.zeros で (高さ, 幅, チャンネル数) の真っ黒な画像配列を作る
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # 2. 1つ目の動画 (frame1) を左上に貼り付け
        #    canvas[Y座標の範囲, X座標の範囲] = 画像
        canvas[0:height1, 0:width1] = frame1
        
        # 3. 2つ目の動画 (frame2) を1つ目の隣に貼り付け
        #    X座標の開始位置が width1 になる
        canvas[0:height2, width1:width1+width2] = frame2
        # --- ★ 変更点ここまで ---

        # 貼り付けが完了したキャンバスを書き込む
        writer.write(canvas)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"{frame_count} フレーム処理完了...")

    # リソースを解放
    cap1.release()
    cap2.release()
    writer.release()
    
    print(f"\n動画の結合が完了しました。")
    print(f"合計 {frame_count} フレームを {output_path} に保存しました。")

# --- 実行 ---
# 実際の動画ファイルパスに置き換えてください
VIDEO_FILE_1 = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\お手本_元動画\oneself_leg_2.mp4"
VIDEO_FILE_2 = r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\植田部長\スクワット_元動作Mvs元動作U\Ueda_origin.mp4"

# 保存したい場所とファイル名を指定
OUTPUT_FILE = r'C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\combined_video.mp4'

combine_videos_with_padding(VIDEO_FILE_1, VIDEO_FILE_2, OUTPUT_FILE)