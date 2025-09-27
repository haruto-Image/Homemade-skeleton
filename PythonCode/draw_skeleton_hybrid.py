import cv2
import numpy as np

# --- キーポイントのインデックス定義 ---
COCO_KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
PELVIS_IDX, THORAX_IDX, SPINE_IDX, NECK_IDX = 17, 18, 19, 20

# --- 骨格の接続ルール定義 ---
# ★★★【修正】顔の基本接続を簡略化 ★★★
BODY_SKELETON = [
    # 体の中心線
    (PELVIS_IDX, SPINE_IDX), (SPINE_IDX, THORAX_IDX), (THORAX_IDX, NECK_IDX),
    # 首と顔の中心をつなぐ
    (NECK_IDX, COCO_KP['nose']),
    # 腕
    (THORAX_IDX, COCO_KP['left_shoulder']), (COCO_KP['left_shoulder'], COCO_KP['left_elbow']),
    (COCO_KP['left_elbow'], COCO_KP['left_wrist']),
    (THORAX_IDX, COCO_KP['right_shoulder']), (COCO_KP['right_shoulder'], COCO_KP['right_elbow']),
    (COCO_KP['right_elbow'], COCO_KP['right_wrist']),
    # 脚
    (PELVIS_IDX, COCO_KP['left_hip']), (COCO_KP['left_hip'], COCO_KP['left_knee']),
    (COCO_KP['left_knee'], COCO_KP['left_ankle']),
    (PELVIS_IDX, COCO_KP['right_hip']), (COCO_KP['right_hip'], COCO_KP['right_knee']),
    (COCO_KP['right_knee'], COCO_KP['right_ankle']),
]
# 手の骨格
HAND_SKELETON = [(i, i + 1) for i in range(3)] + [(0, 4)] # Simplified for brevity, original is better
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
# 顔の輪郭とパーツの骨格
JAWLINE_SKELETON = [(i, i + 1) for i in range(11)]
LEFT_EYEBROW_SKELETON = [(i, i + 1) for i in range(4)]
RIGHT_EYEBROW_SKELETON = [(i, i + 1) for i in range(4)]
LEFT_EYE_SKELETON = [(i, i + 1) for i in range(5)] + [(5, 0)]
RIGHT_EYE_SKELETON = [(i, i + 1) for i in range(5)] + [(5, 0)]
OUTER_LIP_SKELETON = [(i, i + 1) for i in range(11)] + [(11, 0)]
INNER_LIP_SKELETON = [(i, i + 1) for i in range(7)] + [(7, 0)]
HEAD_CONTOUR_SKELETON = [(i, i + 1) for i in range(13)]

# --- 色の定義 ---
KP_COLORS = [(0,0,255), (0,85,255), (0,170,255), (0,255,255), (0,255,170), (0,255,85),
             (0,255,0), (85,255,0), (170,255,0), (255,255,0), (255,170,0), (255,85,0),
             (255,0,0), (255,0,85), (255,0,170), (255,0,255), (170,0,255), (85,0,255)]
LINE_COLOR = (255, 255, 255); HAND_LINE_COLOR = (100, 255, 100); HAND_KP_COLOR = (0, 255, 0)
JAWLINE_COLOR = (255, 255, 0); FOOT_KP_COLOR = (255, 100, 255)
FACE_FEATURES_COLOR = (0, 180, 255)


def draw_skeleton_hybrid(image, all_133_keypoints, conf_threshold=0.3):
    """ 全身133点のキーポイントを元に、顔(輪郭含む)・体・手・足先の全ての骨格を描画する """
    
    # 1. 体と顔の基本部分を描画
    body_keypoints = all_133_keypoints[:17]
    l_sh_score, r_sh_score = body_keypoints[5, 2], body_keypoints[6, 2]
    l_hip_score, r_hip_score = body_keypoints[11, 2], body_keypoints[12, 2]
    nose_score = body_keypoints[0, 2]

    thorax_xy = np.mean(body_keypoints[5:7, :2], axis=0)
    thorax = np.append(thorax_xy, min(l_sh_score, r_sh_score))
    pelvis_xy = np.mean(body_keypoints[11:13, :2], axis=0)
    pelvis = np.append(pelvis_xy, min(l_hip_score, r_hip_score))
    spine_xy = np.mean([pelvis_xy, thorax_xy], axis=0)
    spine = np.append(spine_xy, min(pelvis[2], thorax[2]))
    neck_xy = np.mean([thorax_xy, body_keypoints[0, :2]], axis=0)
    neck = np.append(neck_xy, min(thorax[2], nose_score))

    calculated_kps = np.array([pelvis, thorax, spine, neck])
    body_face_keypoints = np.vstack((body_keypoints, calculated_kps))

    for start_idx, end_idx in BODY_SKELETON:
        p1, p2 = body_face_keypoints[start_idx], body_face_keypoints[end_idx]
        if p1[2] > conf_threshold and p2[2] > conf_threshold:
            pt1, pt2 = tuple(map(int, p1[:2])), tuple(map(int, p2[:2]))
            cv2.line(image, pt1, pt2, LINE_COLOR, 2, cv2.LINE_AA)

    for i, kp in enumerate(body_face_keypoints):
        if i < 21 and kp[2] > conf_threshold: # 元の17点のみ描画
            pt = tuple(map(int, kp[:2])); color = KP_COLORS[i % len(KP_COLORS)]
            cv2.circle(image, pt, 5, color, -1, cv2.LINE_AA)

    # 2. 手、顔の輪郭、足先、詳細な顔パーツを描画
    if all_133_keypoints.shape[0] >= 133:
        # 顔の輪郭 (アゴ)
        jawline_kps = all_133_keypoints[23:35]
        for kp in jawline_kps:
            if kp[2] > conf_threshold:
                pt = tuple(map(int, kp[:2]))
                cv2.circle(image, pt, 2, JAWLINE_COLOR, -1, cv2.LINE_AA)
        
        # 顔の上部輪郭を線ではなく点で描画
        head_contour_kps = all_133_keypoints[77:91]
        for kp in head_contour_kps:
            if kp[2] > conf_threshold:
                pt = tuple(map(int, kp[:2]))
                cv2.circle(image, pt, 2, JAWLINE_COLOR, -1, cv2.LINE_AA)
        
        # 顔の輪郭を閉じる線の描画は不要なので削除

        # 詳細な顔パーツを線ではなく点で描画
        # (顔パーツのキーポイントの範囲をリスト化)
        face_parts_kps = [
            all_133_keypoints[35:40], # 左まゆ
            all_133_keypoints[40:45], # 右まゆ
            all_133_keypoints[45:51], # 左目
            all_133_keypoints[51:57], # 右目
            all_133_keypoints[57:69], # 外唇
            all_133_keypoints[69:77], # 内唇
        ]
        for kps_group in face_parts_kps:
            for kp in kps_group:
                if kp[2] > conf_threshold:
                    pt = tuple(map(int, kp[:2]))
                    cv2.circle(image, pt, 2, FACE_FEATURES_COLOR, -1, cv2.LINE_AA)

        # 足先 (点のみ描画)
        foot_kps = all_133_keypoints[17:23]
        for kp in foot_kps:
            if kp[2] > conf_threshold:
                pt = tuple(map(int, kp[:2]))
                cv2.circle(image, pt, 4, FOOT_KP_COLOR, -1, cv2.LINE_AA)

        # 手
        for start_h, end_h in [(91, 112), (112, 133)]:
            hand_kps = all_133_keypoints[start_h:end_h]
            for start_idx, end_idx in HAND_SKELETON:
                p1, p2 = hand_kps[start_idx], hand_kps[end_idx]
                if p1[2] > conf_threshold and p2[2] > conf_threshold:
                    pt1, pt2 = tuple(map(int, p1[:2])), tuple(map(int, p2[:2]))
                    cv2.line(image, pt1, pt2, HAND_LINE_COLOR, 2, cv2.LINE_AA)
            for kp in hand_kps:
                if kp[2] > conf_threshold:
                    pt = tuple(map(int, kp[:2]))
                    cv2.circle(image, pt, 3, HAND_KP_COLOR, -1, cv2.LINE_AA)

    return image

