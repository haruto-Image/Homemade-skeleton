import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import subprocess

def plot_comparison_graph(args,df1,df2):
    """
    2つのCSVファイル（'frame', 'x', 'y' カラムを持つ想定）を読み込み、
    座標を比較するグラフを生成する。
    """
    # ループ内で使用するため、ファイル名はargsから取得
    csv_path1 = Path(args.csv1)
    csv_path2 = Path(args.csv2)

    target_kp_id = args.plot_kp_id
    
    # 2. 指定されたIDのデータだけを抽出する
    # df1, df2 は main から渡された全データ
    kp_data1 = df1[df1['kp_id'] == target_kp_id]
    kp_data2 = df2[df2['kp_id'] == target_kp_id]

    

    # 必須カラムのチェック
    required_cols = ['frame', 'x', 'y', 'kp_id'] # kp_idもチェック対象に追加
    if not all(col in df1.columns for col in required_cols) or \
       not all(col in df2.columns for col in required_cols):
        print(f"エラー: CSVファイルには 'frame', 'x', 'y', 'kp_id' のカラムが必要です。")
        print(f"  ファイル1のカラム: {list(df1.columns)}")
        print(f"  ファイル2のカラム: {list(df2.columns)}")
        return # このIDの処理をスキップ

    if kp_data1.empty or kp_data2.empty:
        print(f"エラー: キーポイントID {target_kp_id} のデータが見つからないか空です。")
        print(f"  ファイル1 ({csv_path1.name}): {len(kp_data1)} 件")
        print(f"  ファイル2 ({csv_path2.name}): {len(kp_data2)} 件")
        return # このIDの処理をスキップ

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
             
    plt.title(f'Keypoint {target_kp_id} - X Coordinate Comparison (Normalized)')
    plt.xlabel('Frame')
    plt.ylabel('X coordinate (normalized)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # グラフの重なりを防ぐ

    if args.out_graph_dynamic:
        graph_path = Path(args.out_graph_dynamic)
    else:
        graph_path = csv_path1.parent/f'comparison_kp{target_kp_id}_{csv_path1.stem}_vs_{csv_path2.stem}.png'
    
    resolved_path = graph_path.resolve()
    print(f"このパスに保存{resolved_path}")

    try:
        plt.savefig(graph_path)
        print(f"比較グラフ(KP {target_kp_id}) を {resolved_path} に保存しました。")
    except Exception as e:
        # もし保存に失敗したら、エラー内容を表示します
        print(f"!!!!!!! エラー: グラフ (KP {target_kp_id}) の保存に失敗しました !!!!!!!")
        print(f"エラー詳細: {e}")
        print(f"失敗したパス: {resolved_path}")
    
    plt.close()
    


def main():
    ap = argparse.ArgumentParser(description="2つのキーポイントCSVを比較するグラフを生成します。")
    
    ap.add_argument("--csv1", default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\植田部長\スクワット_元動作Mvs元動作U\Mino_leg_shorts_keypoints_10.csv",help="比較グラフ用: 1つ目のキーポイントCSVパス")
    ap.add_argument("--csv2",default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\中村さん\スクワット_元動画Mvs元動画N\Nakamura_legs_keypoints_10.csv",help="比較グラフ用: 2つ目のキーポイントCSVパス")
    
    # --- ★ 変更点 4: --plot_kp_id オプションを削除 ---
    # ap.add_argument("--plot_kp_id", ...)
    ap.add_argument("--out_graph", 
                    default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\骨格関係\生成前後データ比較\植田部長\スクワット_元動作Mvs元動作U\U_比較.png", 
                    help="比較グラフ用: 出力するグラフ画像ファイルパスの「ベース名」。例: 'base.png' -> 'base_kp0.png', 'base_kp1.png'...")
    
    ap.add_argument("--sample_rate", type=int, default=10, help="グラフ描画時のサンプリングレート（データが多い場合）")
    args = ap.parse_args()
    
    print("--- 全キーポイント (0-17) の比較グラフを生成します ---")
    
    # --- ★ 変更点: 出力ファイル名のベースを準備 ---
    original_out_path = None
    out_dir = Path(".")
    out_stem = ""
    out_suffix = ".png"

    if args.out_graph:
        original_out_path = Path(args.out_graph)
        out_dir = original_out_path.parent
        out_stem = original_out_path.stem # ファイル名の拡張子なし(例：U_比較)
        out_suffix = original_out_path.suffix # ファイル名の拡張子

    # --- ★ 変更点: CSVをループ前に一度だけ読み込む ---
    try:
        df1_all = pd.read_csv(args.csv1)
        df2_all = pd.read_csv(args.csv2)
        print(f"読み込み成功 (1): {args.csv1}")
        print(f"読み込み成功 (2): {args.csv2}")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e}")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    # --- ★ 変更点: 0から17までループ ---
    for kp_id in range(18):
        print(f"\n--- キーポイント {kp_id} のグラフを生成中... ---")
        
        # argsにループ中のIDを設定
        args.plot_kp_id = kp_id
        
        # argsに動的な出力パスを設定
        if original_out_path:
            # --out_graph が指定されている場合、ファイル名にIDを挿入
            # 例: U_比較.png -> U_比較_kp0.png
            new_out_stem = f"{out_stem}_kp{kp_id}"
            args.out_graph_dynamic = out_dir / (new_out_stem + out_suffix)
        else:
            # --out_graph が指定されていない場合 (plot_comparison_graph内のロジックで自動生成させる)
            args.out_graph_dynamic = None # type: ignore

        # グラフ描画関数に「全データ」を渡す
        plot_comparison_graph(args, df1_all, df2_all)
        
    #グラフを自動で開く (Windows)
    try:
        if os.name == "nt":
            print(f"保存先フォルダ {out_dir.resolve()} を開きます...")
            subprocess.run(["explorer", str(out_dir.resolve())])
    except Exception as e:
        print(f"グラフを自動で開けませんでした: {e}")

    print("\n--- 全てのグラフ生成が完了しました。 ---")

    #plot_comparison_graph(args)

if __name__ == "__main__":
    main()