import cv2
import math
import numpy as np

BODY_SKELETON = [
    [12, 13], [1,18],[1, 2], [2, 12], [3, 4], [4, 5], [5, 12], [6, 7], 
    [7, 8], [9, 10], [10, 11], [8, 12], [11, 12], [13, 16], [13, 17], 
    [14, 16], [15, 17],
]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
# colors = [[0, 0, 255], [0, 85, 255], [0, 170, 255], [0, 255, 255], [0, 255, 170], [0, 255, 85], 
#           [0, 255, 0], [85, 255, 0], [170, 255, 0], [255, 255, 0], [255, 170, 0], [255, 85, 0], 
#           [255, 0, 0], [255, 0, 85], [255, 0, 170], [255, 0, 255], [170, 0, 255]]
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

LINE_COLOR = (255, 255, 255); HAND_LINE_COLOR = (100, 255, 100); HAND_KP_COLOR = (0, 255, 0)
JAWLINE_COLOR = (255, 255, 0); FOOT_KP_COLOR = (255, 100, 255)
FACE_FEATURES_COLOR = (0, 180, 255)

def is_normalized(all_133_keypoints):
    point_normalized = [
        0 <= abs(k[0]) <= 1 and 0 <= abs(k[1]) <= 1
        for k in all_133_keypoints
    ]
    if not point_normalized:
        return False
    return all(point_normalized)


def draw_skeleton_hybrid(image, all_133_keypoints, conf_threshold=0.3):
    """ 全身133点のキーポイントを元に、顔(輪郭含む)・体・手・足先の全ての骨格を描画する """
    
    body_keypoints = all_133_keypoints[:17] #133個の内戦闘の17個だけ抜き出す
    l_sh_score, r_sh_score = body_keypoints[5, 2], body_keypoints[6, 2]

    thorax_xy = np.mean(body_keypoints[5:7, :2], axis=0)#body_keypoints[5:7, :2]は5番目と6番目のキーポイントの2つの要素(x,y)を抜き出してくる
    thorax = np.append(thorax_xy, min(l_sh_score, r_sh_score))

    calculated_kps = np.array(thorax)
    O_body_face_keypoints = np.vstack((body_keypoints, calculated_kps))#17個のキーポイントの末尾に18個目のthoraxを追加
    new_order = [13, 11, 16, 14, 12, 9, 7, 5, 10, 8, 6, 17, 0, 3, 4, 1, 2, 15]#シャーペンの順序になった
    body_face_keypoints = O_body_face_keypoints[new_order]

    for (k1_index, k2_index), color in zip(BODY_SKELETON , colors):
        if not is_normalized(body_face_keypoints):
            H, W = 1.0, 1.0
        else:
            H, W, _ = image.shape

        CH, CW, _ = image.shape
        stickwidth = 4

        # Ref: https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
        max_side = max(CW, CH)
        stick_scale = 1

        keypoint1 = body_face_keypoints[k1_index - 1]
        keypoint2 = body_face_keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        #正規化で表現するので見やすいようにimage[row, col]はimage[y, x]として表現している
        Y = np.array([keypoint1[0], keypoint2[0]])* float(W)
        X = np.array([keypoint1[1], keypoint2[1]])* float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth*stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(image, polygon, [int(float(c) * 0.6) for c in color])

    for body_face_keypoints, color in zip(body_face_keypoints, colors):
        if body_face_keypoints is None:
            continue

        x, y = body_face_keypoints[0], body_face_keypoints[1]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(image, (int(x), int(y)), 20, color, thickness=-1)

    

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