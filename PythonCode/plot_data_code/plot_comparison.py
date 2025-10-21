import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="2つのキーポイントCSVデータを比較してグラフ化します。")
    parser.add_argument("csv_file1", help="1つ目のCSVファイルのパス")
    parser.add_argument("csv_file2", help="2つ目のCSVファイルのパス")
    parser.add_argument("--kp_id", type=int, default=10, help="グラフ化するキーポイントのID (デフォルト: 10)")
    parser.add_argument("--output_image", default="comparison_graph.png", help="出力するグラフの画像ファイル名")
    args = parser.parse_args()

    # CSVファイルを読み込む
    try:
        df1 = pd.read_csv(args.csv_file1)
        df2 = pd.read_csv(args.csv_file2)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    # 指定されたキーポイントIDのデータを抽出
    kp_data1 = df1[df1['kp_id'] == args.kp_id]
    kp_data2 = df2[df2['kp_id'] == args.kp_id]

    # --- グラフ描画 ---
    plt.figure(figsize=(15, 7))

    # 1つ目のデータをプロット
    if not kp_data1.empty:
        plt.plot(kp_data1['frame'], kp_data1['y'], label=f'Video 1 (Y coord)', color='blue')

    # 2つ目のデータをプロット
    if not kp_data2.empty:
        # 比較しやすいように、2つ目のデータは破線にする
        plt.plot(kp_data2['frame'], kp_data2['y'], label=f'Video 2 (Y coord)', color='red', linestyle='--')

    # グラフの装飾
    plt.title(f'Comparison of Keypoint ID: {args.kp_id} (Y-axis Trajectory)')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Y-Coordinate (0.0 - 1.0)')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1) # Y軸の範囲を0から1に固定すると比較しやすい

    # グラフをファイルに保存
    save_path = Path(args.output_image)
    plt.savefig(save_path)
    print(f"比較グラフを {save_path} に保存しました。")

if __name__ == "__main__":
    main()