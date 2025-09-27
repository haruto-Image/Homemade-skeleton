import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse,os
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
import comfy_dwpose_core as DWPoseCore
# 同じフォルダにあるutil.pyから機能を読み込む
from typing import Tuple, List
from draw_skeleton_hybrid_test import draw_skeleton_hybrid
from OneEuroFilter import OneEuroFilter
#from dwpose_core import DWPoseCore

# --------------------
# ユーティリティ
# --------------------
def pick_hw_from_model(sess):
    # 入力テンソル shape: (N, C, H, W) 想定。動的軸は None or -1 のことがある
    s = sess.get_inputs()[0].shape
    # 末尾2つが H,W になるように解釈
    H, W = int(s[-2]) if s[-2] not in (None, 'None', -1) else None, int(s[-1]) if s[-1] not in (None, 'None', -1) else None
    return H, W
    #この部分のおかげでAIモデルが処理できる画像のサイズをいちいち指定しなくても使用するモデルに合わせたサイズに調整してくれる

def letterbox(img, new_shape, color=(114,114,114)):
    #Yoloモデルがbboxを検出しやすいように事前に画像を整える(Yoloが受け取れる固定サイズにリサイズ)もの
    #どんなサイズの入力動画が来てもリサイズしてそのうえで人物を探し、元の大きさに戻す
    h, w = img.shape[:2]
    H, W = new_shape
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad = (W - nw, H - nh)
    top = pad[1] // 2; bottom = pad[1] - top
    left = pad[0] // 2; right = pad[0] - left
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def nms(boxes, scores, iou_thr=0.45, top_k=300):
    #すべての検出ボックスをならべ、そのボックスの評価をつけてどれを採用するか決定するもの
    # boxes: Nx4 (x1,y1,x2,y2), scores: N
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < top_k:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w*h
        iou = inter / ( (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) + (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1]) - inter + 1e-6 )
        idxs = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int32)

# --------------------
# YOLOX（ONNX）: 人物検出
# --------------------

class YOLOXDetector:
    def __init__(self, onnx_path, providers):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.Hin, self.Win = pick_hw_from_model(self.sess)
        if self.Hin is None or self.Win is None:
            self.Hin, self.Win = 640, 640
        self.input_name = self.sess.get_inputs()[0].name

    def _decode_output(self, output):
        if output.ndim == 3: output = output[0]
        strides = [8, 16, 32]
        grids, expanded_strides = [], []
        for i, stride in enumerate(strides):
            h, w = self.Hin // stride, self.Win // stride
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), stride))
        grids = np.concatenate(grids, axis=0)
        expanded_strides = np.concatenate(expanded_strides, axis=0)
        output[..., :2] = (output[..., :2] + grids) * expanded_strides
        output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides
        return output

    def infer_person_boxes(self, img_bgr, conf_thr=0.05, iou_thr=0.45):
        # letterboxのダミー実装が Pillow を使うため、numpy -> Pillow Imageに変換    
        inp_pil, r, (padw, padh) = letterbox(img_bgr, (self.Hin, self.Win))
        inp = np.array(inp_pil)

        img_rgb = inp.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std
        blob = norm_img.transpose(2,0,1)
        blob = np.expand_dims(blob, 0)

        outs = self.sess.run(None, {self.input_name: blob})
        out = self._decode_output(outs[0])

        if out.ndim == 3: out = out[0]
        
        if out.shape[1] >= 85: # COCO model
            xywh = out[:, :4]
            obj  = out[:, 4]
            cls  = out[:, 5:]
            scores = obj * cls[:, 0] # person class
            x, y, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
            x1 = (x - w/2); y1 = (y - h/2); x2 = (x + w/2); y2 = (y + h/2)
            boxes = np.stack([x1,y1,x2,y2], axis=1)
        else: # custom model
            boxes = out[:, :4]
            scores = out[:, 4]

        if boxes.size == 0:
            return np.empty((0,4), np.float32), np.empty((0,), np.float32)

        boxes[:, [0,2]] -= padw
        boxes[:, [1,3]] -= padh
        boxes /= r

        x1 = np.minimum(boxes[:,0], boxes[:,2])
        y1 = np.minimum(boxes[:,1], boxes[:,3])
        x2 = np.maximum(boxes[:,0], boxes[:,2])
        y2 = np.maximum(boxes[:,1], boxes[:,3])
        boxes = np.stack([x1,y1,x2,y2], axis=1)

        H, W = img_bgr.shape[:2]
        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, W-1)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, H-1)

        m = scores > conf_thr
        boxes, scores = boxes[m], scores[m]
        if boxes.size == 0:
            return np.empty((0,4), np.float32), np.empty((0,), np.float32)

        keep = nms(boxes, scores, iou_thr=iou_thr)
        final_boxes = boxes[keep]
        final_scores = scores[keep]

        if final_boxes.size == 0:
            return np.empty((0,4), np.float32), np.empty((0,), np.float32)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ ここからが修正箇所 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        
        H, W = img_bgr.shape[:2]

        # 元のBBoxの幅と高さを計算
        box_widths = final_boxes[:, 2] - final_boxes[:, 0]
        box_heights = final_boxes[:, 3] - final_boxes[:, 1]
        
        # 1. BBoxの下端を強制的に画像の下端まで拡張
        #    これが全身を捉えるための最も重要なステップです。
        final_boxes[:, 3] = H - 1

        # 2. BBoxを左右と上に拡張してマージンを持たせる
        #    BBoxのサイズに応じた割合で拡張することで、より頑健になります。
        #    ここでは幅の10%、高さの10%をマージンとして追加します。
        horizontal_margin = box_widths * 0.1
        vertical_margin = box_heights * 0.1

        final_boxes[:, 0] -= horizontal_margin  # x1 (左)
        final_boxes[:, 2] += horizontal_margin  # x2 (右)
        final_boxes[:, 1] -= vertical_margin    # y1 (上)
        
        # 3. 拡張したBBoxが画像サイズを超えないようにクリップする
        final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]].clip(0, W - 1)
        final_boxes[:, [1, 3]] = final_boxes[:, [1, 3]].clip(0, H - 1)

        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ここまでが修正箇所 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        return final_boxes.astype(np.float32), final_scores.astype(np.float32)


# --------------------
# DWPose（ONNX）: 単人体キーポイント
# --------------------
class DWPoseRunner:
    def __init__(self, onnx_path, providers):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        # モデルの入力サイズを取得または設定
        s = self.sess.get_inputs()[0].shape
        self.Hin = int(s[-2]) if s[-2] not in (None, 'None', -1) else 288
        self.Win = int(s[-1]) if s[-1] not in (None, 'None', -1) else 384
        self.input_name = self.sess.get_inputs()[0].name

    # DWPoseRunner クラスの中
    def infer_keypoints133(self, img_bgr, box):
        # ...（前処理と推論の部分は、これまで通りで変更なし）...
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_bbox = [np.array(box)]
        model_input_size = (self.Win, self.Hin)
        resized_imgs, centers, scales = DWPoseCore.preprocess(img_rgb, out_bbox, model_input_size)
        resized_img, center, scale = resized_imgs[0], centers[0], scales[0]
        inp = resized_img.astype(np.float32).transpose(2, 0, 1)
        inp = np.expand_dims(inp, axis=0)
        outputs = self.sess.run(None, {self.input_name: inp})
        all_keypoints, all_scores = DWPoseCore.postprocess([outputs], [model_input_size], [center], [scale])
        
        if all_keypoints is None or len(all_keypoints) == 0:
            return None
            
        keypoints = all_keypoints[0]
        scores = all_scores[0]
        keypoints_info = np.concatenate((keypoints, scores[:, np.newaxis]), axis=1)

        # ▼▼▼【ここからが追加する並べ替えロジック】▼▼▼
        # 元のキーポイントは133点
        # ComfyUIの実装では、首(neck)の点を計算して追加し、134点として扱っている
        
        # 首の座標を計算 (5番:左肩 と 6番:右肩 の中間)
        neck = np.mean(keypoints_info[[5, 6]], axis=0)
        # 首のスコアを計算 (両肩のスコアが0.3以上なら1.0、そうでなければ0.0)
        neck_score = 1.0 if keypoints_info[5, 2] > 0.3 and keypoints_info[6, 2] > 0.3 else 0.0
        neck[2] = neck_score
        
        # 元の133点に首を追加して134点にする
        # 17番の位置に挿入する
        # keypoints_info = np.insert(keypoints_info, 17, neck, axis=0)
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]
        
        keypoints_info = np.concatenate((keypoints, scores), axis=1)

        return keypoints_info.astype(np.float32)

# --------------------
# 133点 → Body25風 & 可視化
# --------------------

# OpenPose Body25の標準的な接続順
EDGES = [
    # 顔
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 17),
    # 体幹
    (17, 5), (17, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
    # 脚
    (11, 13), (12, 14), (13, 15), (14, 16)
]

def draw_skeleton(canvas, kps25, edges=EDGES, thr=0.01):
    drawn = 0
    for a,b in edges:
        pa, pb = kps25[a], kps25[b]
        if pa[2] > thr and pb[2] > thr:
            cv2.line(canvas, tuple(pa[:2].astype(int)), tuple(pb[:2].astype(int)), (255,255,255), 2)
        for p in kps25:
            cv2.circle(canvas, tuple(p[:2].astype(int)), 2, (0,0,255), -1)
    return canvas


# --------------------
# メイン
# --------------------
def main():
    ap = argparse.ArgumentParser()
    # 既定パス（必要に応じて書き換え）
    ap.add_argument("--video", default=r"C:\Users\_s2520798\Documents\1.研究\入出力映像\input\0908(お手本動画の画質検証)\800_1200\Mino_leg_shorts.mp4")
    ap.add_argument("--det",   default=r"C:\Users\_s2520798\Documents\1.研究\動画編集python\models\yolox_l.onnx")
    ap.add_argument("--pose",  default=r"C:\Users\_s2520798\Documents\1.研究\動画編集python\models\dw-ll_ucoco_384.onnx")

    # 出力先
    ap.add_argument("--out",       default=r"C:\Users\_s2520798\Documents", help="PNG/動画の出力フォルダ")
    ap.add_argument("--out_video", default=None, help="出力動画のフルパス（未指定なら out/pose_out1.mp4）")

    # 表示・保存オプション ←★これが無いと AttributeError
    ap.add_argument("--overlay",  action="store_true", help="元映像に骨格を重ねる（指定しなければ黒背景）")
    ap.add_argument("--save_png", action="store_true", help="動画に加えてPNG連番も保存する")

    # 実行制御
    ap.add_argument("--cpu",   action="store_true", help="CPUのみで実行（CUDA警告を抑制）")
    ap.add_argument("--conf",  type=float, default=0.05, help="検出スコア閾値")
    ap.add_argument("--iou",   type=float, default=0.45, help="NMS IOU閾値")
    ap.add_argument("--every", type=int,   default=1,    help="何フレームごとに処理するか（間引き）")

    # スムージング処理の追加
    ap.add_argument("--mincutoff", type=float, default=0.0005, help="OneEuroFilter: mincutoff (小さいほど強く平滑化)")
    ap.add_argument("--beta",       type=float, default=0.7,   help="OneEuroFilter: beta (大きいほど高速な動きに追従)")


    args = ap.parse_args()  #この設定以降args.outのようにするだけでパスを呼び出せる

    # --- 準備フェーズ ---
    # AIモデルの推論をどのハードで実行するかを指定
    providers = ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # 人物検出と姿勢推定を行うクラスのインスタンスを生成（ループの前に一度だけ）
    det = YOLOXDetector(args.det, providers)
    pose = DWPoseRunner(args.pose, providers) # poseもここで準備する

    # 動画の読み込み
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {args.video}")

    # 出力動画の設定
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = max(1.0, fps_in / max(1, args.every))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = Path(args.out_video) if args.out_video else (out_dir / "pose_out1.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_video_path), fourcc, fps_out, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriterを開けません: {out_video_path}")
    
    num_keypoints = 133
    filters_xy = [
        [OneEuroFilter(freq=fps_in, mincutoff=args.mincutoff, beta=args.beta), # x用フィルター
         OneEuroFilter(freq=fps_in, mincutoff=args.mincutoff, beta=args.beta)] # y用フィルター
        for _ in range(num_keypoints)
    ]

    i = 0
    saved = 0

    # --- 処理ループフェーズ ---
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if (i % args.every) != 0:
            i += 1
            continue

        canvas = frame.copy() if args.overlay else np.zeros_like(frame)
        # 1) 人物検出
        boxes, scores = det.infer_person_boxes(frame, conf_thr=args.conf, iou_thr=args.iou)
        
        # 検出されなかった場合はスキップ
        if boxes.shape[0] == 0:
            print(f"フレーム {i}: 人物が検出されませんでした。スキップします。")
            # 黒背景の場合はそのまま黒フレームを出力、オーバーレイの場合は元映像をそのまま出力
            canvas = np.zeros_like(frame) if not args.overlay else frame
            vw.write(canvas)
            i += 1
            saved += 1
            continue

        # 一番大きい人を対象にする
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = np.argmax(areas)
        box = boxes[best_idx]

        # 2) DWPoseの姿勢推定（ComfyUI版ロジック）
        keypoints_info = pose.infer_keypoints133(frame, box)

        
        if keypoints_info is None:
            print("キーポイントの検出に失敗しました。このフレームをスキップします。")
            i += 1
            continue

        smoothed_keypoints = np.zeros_like(keypoints_info)

        for kp_idx in range(num_keypoints):
            # 元の座標と信頼度を取得
            x, y, conf = keypoints_info[kp_idx]
            
            # フィルターを適用してx, y座標を平滑化
            # 信頼度が低い点も一度フィルターに通すことで、急な座標の跳びを抑制する
            smoothed_x = filters_xy[kp_idx][0](x, None)
            smoothed_y = filters_xy[kp_idx][1](y, None)

            # 平滑化した座標と元の信頼度を格納
            smoothed_keypoints[kp_idx] = [smoothed_x, smoothed_y, conf]

        # 3) 骨格の描画
        canvas = draw_skeleton_hybrid(canvas, smoothed_keypoints, conf_threshold=0.3)

        # 4) 動画に書き込み
        vw.write(canvas)
        saved += 1
        i += 1
        print(f"フレーム {i-1} を処理しました。")

    vw.release()
    print(f"完了しました。動画を {out_video_path} に保存しました。")
    print(f"処理フレーム数: {saved} (出力FPS={fps_out})")
    
    # フォルダ/ファイルを開く（Windows）
    try:
        import subprocess
        if os.name == "nt":
            subprocess.run(["explorer", "/select,", str(out_video_path)])
    except Exception as e:
        print(f"フォルダを自動で開けませんでした: {e}")


if __name__ == "__main__":
    main()
