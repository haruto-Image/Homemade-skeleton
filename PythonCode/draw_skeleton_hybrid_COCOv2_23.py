import cv2
import numpy as np

# --- 定数定義 ---

# DWPoseから取得するCOCO互換キーポイントのインデックス定義
COCO_KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
THORAX_IDX = 17

# COCO(18点)形式の骨格を描画するための、点と点を結ぶ線のルール
COCO_SKELETON = [
    (COCO_KP['nose'], COCO_KP['left_eye']), (COCO_KP['left_eye'], COCO_KP['left_ear']),
    (COCO_KP['nose'], COCO_KP['right_eye']), (COCO_KP['right_eye'], COCO_KP['right_ear']),
    (COCO_KP['left_shoulder'], THORAX_IDX),
    (THORAX_IDX, COCO_KP['right_shoulder']),
    (THORAX_IDX, COCO_KP['left_hip']),
    (THORAX_IDX , COCO_KP['right_hip']),
    (THORAX_IDX, COCO_KP['nose']),
    (COCO_KP['left_shoulder'], COCO_KP['left_elbow']), (COCO_KP['left_elbow'], COCO_KP['left_wrist']),
    (COCO_KP['right_shoulder'], COCO_KP['right_elbow']), (COCO_KP['right_elbow'], COCO_KP['right_wrist']),
    (COCO_KP['left_hip'], COCO_KP['left_knee']), (COCO_KP['left_knee'], COCO_KP['left_ankle']),
    (COCO_KP['right_hip'], COCO_KP['right_knee']), (COCO_KP['right_knee'], COCO_KP['right_ankle'])
]

# 描画に使う色のリスト
KP_COLORS = [(0,0,255), (0,85,255), (0,170,255), (0,255,255), (0,255,170), (0,255,85),
             (0,255,0), (85,255,0), (170,255,0), (255,255,0), (255,170,0), (255,85,0),
             (255,0,0), (255,0,85), (255,0,170), (255,0,255), (170,0,255), (85,0,255)]


# --- 関数1：データ変換の専門家 ---

def create_coco17_stabilized(all_133_keypoints, conf_threshold=0.3):
    """
    DWPoseの133点から、COCO(17点)形式の骨格を生成する。
    その際、MPIIの中心線の考え方を「ガイド」として利用し、主要関節点（肩、腰）を安定化させる。
    """
    # DWPoseの基本17点（COCO互換）をコピーして、これを補正していく。~17まで取り出すという意味
    coco_kps = all_133_keypoints[:17].copy()

    # --- MPIIの考え方で、安定した中心点を計算 ---
    l_sh = coco_kps[COCO_KP['left_shoulder']]
    r_sh = coco_kps[COCO_KP['right_shoulder']]
    l_hip = coco_kps[COCO_KP['left_hip']]
    r_hip = coco_kps[COCO_KP['right_hip']]

    pelvis_xy, pelvis_score = (0, 0), 0
    if l_hip[2] > 0.1 and r_hip[2] > 0.1: #DWPoseで算出したヒートマップによる骨盤の信頼度が一定以上ならこの処理を行うという意味
        pelvis_xy = np.mean([l_hip[:2], r_hip[:2]], axis=0)
        pelvis_score = np.mean([l_hip[2], r_hip[2]])
    
    thorax_xy, thorax_score = (0, 0), 0
    if l_sh[2] > 0.1 and r_sh[2] > 0.1:
        thorax_xy = np.mean([l_sh[:2], r_sh[:2]], axis=0)
        thorax_score = np.append(thorax_xy, min([l_sh[2], r_sh[2]]))

    # --- 計算した中心点をガイドに、COCOの主要関節点を「安定化」させる ---
    if pelvis_score > conf_threshold:
        stable_hip_y = pelvis_xy[1]
        coco_kps[COCO_KP['left_hip'], 1] = stable_hip_y #左腰のデータのうち0,1,2の１であるy座標の値を変更している
        coco_kps[COCO_KP['right_hip'], 1] = stable_hip_y

    thorax_keypoint = np.array(thorax_score)
    coco_kps = np.vstack([coco_kps,thorax_keypoint])

    return coco_kps


# --- 関数2：可視化の専門家 ---

def draw_coco17(canvas, keypoints, conf_threshold=0.3):
    """
    COCO(17点)形式のキーポイントを受け取り、骨格を画像に描画する。
    """
    # 線（骨格）を描画
    for p1_idx, p2_idx in COCO_SKELETON:
        p1 = keypoints[p1_idx]
        p2 = keypoints[p2_idx]
        if p1[2] > conf_threshold and p2[2] > conf_threshold:
            pt1 = tuple(map(int, p1[:2]))
            pt2 = tuple(map(int, p2[:2]))
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2) # 白い線

    # 点（関節）を描画
    for i, kp in enumerate(keypoints):
        if kp[2] > conf_threshold:
            pt = tuple(map(int, kp[:2]))
            color = KP_COLORS[i % len(KP_COLORS)]
            cv2.circle(canvas, pt, 5, color, -1, cv2.LINE_AA)

    return canvas