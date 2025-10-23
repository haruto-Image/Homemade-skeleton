import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import subprocess

def plot_comparison_graph(args):
    """
    2つのCSVファイル（'frame', 'x', 'y' カラムを持つ想定）を読み込み、
    座標を比較するグラフを生成する。
    """
    print("--- 比較グラフ生成モード ---")
    
    # 引数のチェック
    if not args.csv1 or not args.csv2:
        print("エラー: --csv1 と --csv2 の両方を指定してください。")
        return
        
    csv_path1 = Path(args.csv1)
    csv_path2 = Path(args.csv2)

    # CSVファイルの読み込み
    try:
        # 読み込んだデータをそのままグラフ化対象とする
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        print(f"読み込み成功 (1): {csv_path1}")
        print(f"読み込み成功 (2): {csv_path2}")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e}")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    target_kp_id = args.plot_kp_id
    
    # 2. 指定されたIDのデータだけを抽出する
    kp_data1 = df1[df1['kp_id'] == target_kp_id]
    kp_data2 = df2[df2['kp_id'] == target_kp_id]

    # 必須カラムのチェック
    required_cols = ['frame', 'x', 'y']
    if not all(col in kp_data1.columns for col in required_cols) or \
       not all(col in kp_data2.columns for col in required_cols):
        print(f"エラー: CSVファイルには 'frame', 'x', 'y' のカラムが必要です。")
        print(f"  ファイル1のカラム: {list(kp_data1.columns)}")
        print(f"  ファイル2のカラム: {list(kp_data2.columns)}")
        return

    if kp_data1.empty or kp_data2.empty:
        print(f"エラー: 一方または両方のCSVファイルが空です。")
        print(f"  ファイル1 ({csv_path1.name}): {len(kp_data1)} 件")
        print(f"  ファイル2 ({csv_path2.name}): {len(kp_data2)} 件")
        return

    plt.figure(figsize=(15, 10))
    
    # サンプリングレートを決定
    sample_rate = args.sample_rate
    if len(kp_data1) < 200 or len(kp_data2) < 200:
        sample_rate = 1 # データが少ない場合はサンプリングしない

    # グラフ1: Y座標の比較 (2行1列の上段)
    plt.subplot(2, 1, 1) 
    plt.plot(kp_data1['frame'][::sample_rate], kp_data1['y'][::sample_rate], 
             linestyle='-', marker='.', markersize=3, alpha=0.8,
             label=f'origin (Y) - {csv_path1.name}')
    plt.plot(kp_data2['frame'][::sample_rate], kp_data2['y'][::sample_rate], 
             linestyle='--', marker='x', markersize=3, alpha=0.8,
             label=f'oneself (Y) - {csv_path2.name}')
    
    # --- ★ 変更点 2: タイトルからID関連のテキストを削除 ---
    plt.title(f'Keypoint {target_kp_id} - Y Coordinate Comparison (Normalized)')
    plt.ylabel('Y coordinate (normalized)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # グラフ2: X座標の比較 (2行1列の下段)
    plt.subplot(2, 1, 2) 
    plt.plot(kp_data1['frame'][::sample_rate], kp_data1['x'][::sample_rate], 
             linestyle='-', marker='.', markersize=3, alpha=0.8,
             label=f'origin (X) - {csv_path1.name}')
    plt.plot(kp_data2['frame'][::sample_rate], kp_data2['x'][::sample_rate], 
             linestyle='--', marker='x', markersize=3, alpha=0.8,
             label=f'oneself (X) - {csv_path2.name}')
             
    # --- ★ 変更点 2: タイトルからID関連のテキストを削除 ---
    plt.title(f'Keypoint {target_kp_id} - X Coordinate Comparison (Normalized)')
    plt.xlabel('Frame')
    plt.ylabel('X coordinate (normalized)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # グラフの重なりを防ぐ

    # グラフの保存先
    if args.out_graph:
        graph_path = Path(args.out_graph)
    else:
        # --- ★ 変更点 3: 保存ファイル名からID関連のテキストを削除 ---
        graph_path = csv_path1.parent / f'comparison_kp{target_kp_id}_{csv_path1.stem}_vs_{csv_path2.stem}.png'
        
    plt.savefig(graph_path)
    print(f"比較グラフを {graph_path.resolve()} に保存しました。")
    
    # グラフを自動で開く (Windows)
    # try:
    #     if os.name == "nt":
    #         subprocess.run(["explorer", str(graph_path.resolve())])
    # except Exception as e:
    #     print(f"グラフを自動で開けませんでした: {e}")

def main():
    ap = argparse.ArgumentParser(description="2つのキーポイントCSVを比較するグラフを生成します。")
    
    ap.add_argument("--csv1", default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\Mino_leg_shorts_keypoints_10.csv",help="比較グラフ用: 1つ目のキーポイントCSVパス")
    ap.add_argument("--csv2",default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\三野_自作骨格_スクワット_keypoints_10.csv",help="比較グラフ用: 2つ目のキーポイントCSVパス")
    
    # --- ★ 変更点 4: --plot_kp_id オプションを削除 ---
    # ap.add_argument("--plot_kp_id", ...)
    target = 17
    ap.add_argument("--plot_kp_id", type=int, default=target, help="比較グラフ用: 対象とするキーポイントID (デフォルト: 10)")
    ap.add_argument("--out_graph",  default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\17_比較.png", help="比較グラフ用: 出力するグラフ画像ファイルパス (未指定なら自動生成)")
    ap.add_argument("--sample_rate", type=int, default=10, help="グラフ描画時のサンプリングレート（データが多い場合）")

    args = ap.parse_args()
    
    plot_comparison_graph(args)

if __name__ == "__main__":
    main()