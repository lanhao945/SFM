# -*- coding:utf-8 -*-
# Time      :2023/6/25 14:29
# Author    :LanHao
# ▄▄▄█████▓ █    ██  ██ ▄█▀▓█████ ▓█████▄
# ▓  ██▒ ▓▒ ██  ▓██▒ ██▄█▒ ▓█   ▀ ▒██▀ ██▌
# ▒ ▓██░ ▒░▓██  ▒██░▓███▄░ ▒███   ░██   █▌
# ░ ▓██▓ ░ ▓▓█  ░██░▓██ █▄ ▒▓█  ▄ ░▓█▄   ▌
#   ▒██▒ ░ ▒▒█████▓ ▒██▒ █▄░▒████▒░▒████▓
#   ▒ ░░   ░▒▓▒ ▒ ▒ ▒ ▒▒ ▓▒░░ ▒░ ░ ▒▒▓  ▒
#     ░    ░░▒░ ░ ░ ░ ░▒ ▒░ ░ ░  ░ ░ ▒  ▒
#   ░       ░░░ ░ ░ ░ ░░ ░    ░    ░ ░  ░
#             ░     ░  ░      ░  ░   ░
#                                  ░

"""

在改模块下重构revise_v2 中的代码

"""

import cv2
import math

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.optimize import least_squares

from . import datas


##########################
# 两张图之间的特征提取及匹配
##########################

def extract_features_v2(images: datas.ImageDataset, sift_obj: cv2.SIFT = None):
    if sift_obj is None:
        sift_obj = cv2.SIFT_create(0, 3, 0.04, 10)

    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []

    for image in images:
        if image is None:
            continue
        key_points, descriptor = sift_obj.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

        if len(key_points) <= 10:
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image[int(p[1])][int(p[0])]
        colors_for_all.append(colors)
    return key_points_for_all, descriptor_for_all, colors_for_all


extract_features = extract_features_v2


def match_features(query, train, camera: datas.Camera):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(query, train, k=2)
    matches = []
    # Apply Lowe's SIFT matching ratio test(MRT)，值得一提的是，这里的匹配没有
    # 标准形式，可以根据需求进行改动。
    for m, n in knn_matches:
        if m.distance < camera.mrt * n.distance:
            matches.append(m)

    return np.array(matches)


def match_all_features(descriptor_for_all, camera: datas.Camera):
    matches_for_all = []
    for i in range(len(descriptor_for_all) - 1):
        matches = match_features(descriptor_for_all[i],
                                 descriptor_for_all[i + 1], camera)
        matches_for_all.append(matches)
    return matches_for_all


######################
# 寻找图与图之间的对应相机旋转角度以及相机平移
######################
def find_transform(k, p1, p2):
    focal_length = 0.5 * (k[0, 0] + k[1, 1])
    principle_point = (k[0, 2], k[1, 2])
    e, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point,
                                   cv2.RANSAC, 0.999, 1.0)
    camera_matrix = np.array([[focal_length, 0, principle_point[0]],
                             [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, r, t, mask = cv2.recoverPose(e, p1, p2, camera_matrix, mask)

    return r, t, mask


def get_matched_points(p1, p2, matches):
    src_pts = np.asarray([p1[m.queryIdx].pt for m in matches])
    dst_pts = np.asarray([p2[m.trainIdx].pt for m in matches])

    return src_pts, dst_pts


def get_matched_colors(c1, c2, matches):
    color_src_pts = np.asarray([c1[m.queryIdx] for m in matches])
    color_dst_pts = np.asarray([c2[m.trainIdx] for m in matches])

    return color_src_pts, color_dst_pts


# 选择重合的点
def mask_out_points(p1, mask):
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])

    return np.array(p1_copy)


def init_structure(k, key_points_for_all, colors_for_all, matches_for_all):
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1],
                                matches_for_all[0])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1],
                                matches_for_all[0])

    if find_transform(k, p1, p2):
        r, t, mask = find_transform(k, p1, p2)
    else:
        r, t, mask = np.array([]), np.array([]), np.array([])

    p1 = mask_out_points(p1, mask)
    p2 = mask_out_points(p2, mask)
    colors = mask_out_points(c1, mask)
    # 设置第一个相机的变换矩阵，即作为剩下摄像机矩阵变换的基准。
    r0 = np.eye(3, 3)
    t0 = np.zeros((3, 1))
    structure = reconstruct(k, r0, t0, r, t, p1, p2)
    rotations = [r0, r]
    motions = [t0, t]
    correspond_struct_idx = []
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) * - 1)
    correspond_struct_idx = correspond_struct_idx
    idx = 0
    matches = matches_for_all[0]
    for i, match in enumerate(matches):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(match.queryIdx)] = idx
        correspond_struct_idx[1][int(match.trainIdx)] = idx
        idx += 1
    return structure, correspond_struct_idx, colors, rotations, motions


#############
# 三维重建
#############
def reconstruct(k, r1, t1, r2, t2, p1, p2):
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(r1)
    proj1[:, 3] = np.float32(t1.T)
    proj2[0:3, 0:3] = np.float32(r2)
    proj2[:, 3] = np.float32(t2.T)
    fk = np.float32(k)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    s = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)
    structure = []

    for i in range(len(s[0])):
        col = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])

    return np.array(structure)


###########################
# 将已作出的点云进行融合
###########################
def fusion_structure(matches, struct_indices, next_struct_indices, structure,
                     next_structure, colors, next_colors):
    for i, match in enumerate(matches):
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        colors = np.append(colors, [next_colors[i]], axis=0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(
            structure) - 1
    return struct_indices, next_struct_indices, structure, colors


# 制作图像点以及空间点
def get_obj_points_and_img_points(matches, struct_indices, structure,
                                  key_points):
    object_points = []
    image_points = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        struct_idx = struct_indices[query_idx]
        if struct_idx < 0:
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)

    return np.array(object_points), np.array(image_points)


########################
# bundle adjustment
########################


def get_3d_pos_v1(pos, ob, r, t, k, camera: datas.Camera):
    p, j = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, k, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > camera.x or abs(e[1]) > camera.y:
        return None
    return pos


def bundle_adjustment(rotations, motions, k, correspond_struct_idx,
                      key_points_for_all, structure, camera: datas.Camera):
    for i in range(len(rotations)):
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r
    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points = key_points_for_all[i]
        r = rotations[i]
        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3d_pos_v1(structure[point3d_id], key_points[j].pt,
                                      r, t, k, camera)
            structure[point3d_id] = new_point

    return structure


#######################
# 作图
#######################


def rebuild(sfm_data: datas.SFMData) -> datas.ColorPoints:
    # K是摄像头的参数矩阵
    k = sfm_data.camera.k

    key_points_for_all, descriptor_for_all, colors_for_all = extract_features(
        sfm_data.image)
    matches_for_all = match_all_features(descriptor_for_all, sfm_data.camera)
    structure, correspond_struct_idx, colors, \
        rotations, motions = init_structure(k, key_points_for_all,
                                            colors_for_all, matches_for_all)

    for i in range(1, len(matches_for_all)):
        object_points, image_points = get_obj_points_and_img_points(
            matches_for_all[i], correspond_struct_idx[i], structure,
            key_points_for_all[i + 1])
        # 在python的opencv中solvePnPRansac函数的第一个参数长度需要大于7，否则会报错
        # 这里对小于7的点集做一个重复填充操作，即用点集中的第一个点补满7个
        if len(image_points) < 7:
            while len(image_points) < 7:
                object_points = np.append(object_points, [object_points[0]],
                                          axis=0)
                image_points = np.append(image_points, [image_points[0]],
                                         axis=0)

        _, r, t, _ = cv2.solvePnPRansac(object_points, image_points, k,
                                        np.array([]))
        r, _ = cv2.Rodrigues(r)
        rotations.append(r)
        motions.append(t)
        p1, p2 = get_matched_points(key_points_for_all[i],
                                    key_points_for_all[i + 1],
                                    matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1],
                                    matches_for_all[i])
        next_structure = reconstruct(k, rotations[i], motions[i], r, t, p1, p2)

        correspond_struct_idx[i], correspond_struct_idx[
            i + 1], structure, colors = fusion_structure(matches_for_all[i],
                                                         correspond_struct_idx[
                                                             i],
                                                         correspond_struct_idx[
                                                             i + 1], structure,
                                                         next_structure,
                                                         colors, c1)
    structure = bundle_adjustment(rotations, motions, k, correspond_struct_idx,
                                  key_points_for_all, structure,
                                  sfm_data.camera)
    i = 0
    # 由于经过bundle_adjustment的structure，会产生一些空的点（实际代表的意思是已被删除）
    # 这里删除那些为空的点
    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors = np.delete(colors, i, 0)
            i -= 1
        i += 1

    return datas.ColorPoints(
        points=structure,
        colors=colors,
    )


if __name__ == '__main__':
    pass
