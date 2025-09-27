import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# 1. 確認したい.pthファイルのパスを指定してください
# 例: 'C:/Users/YourUser/Downloads/Moore-AnimateAnyone/pose_guider.pth'
file_path = r'C:\Users\_s2520798\Downloads\motion_module.pth'

# --- 実行 ---
try:
    # 2. ファイルを読み込む
    # map_location='cpu' を指定すると、GPUがない環境でも安全に読み込めます
    data = torch.load(file_path, map_location='cpu')

    print(f"ファイル '{file_path}' の読み込みに成功しました。")
    print("-" * 30)

    # 3. 中身が辞書形式（state_dict）か確認
    if isinstance(data, dict):
        print("ファイルは state_dict（辞書形式）のようです。")
        print("含まれているキー（層の名前など）の一覧:")
        
        # 4. キーをすべて表示
        key_list = list(data.keys())
        for i, key in enumerate(key_list):
            print(f"  {i+1}: {key}")

        # 5. (おまけ) 最初のキーの重みの形状（サイズ）を表示
        if key_list:
            first_key = key_list[0]
            shape = data[first_key].shape
            print("-" * 30)
            print(f"最初のキー '{first_key}' のパラメータ形状（サイズ）: {shape}")

    else:
        # state_dictではなく、モデルの構造全体が保存されている場合
        print("ファイルはモデル全体のようです。")
        print("モデルの構造:")
        print(data)

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
except Exception as e:
    print(f"エラー: ファイルの読み込み中に問題が発生しました。 {e}")