import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import yaml
import random
import torch
import torchvision.utils as vutils
from pathlib import Path

import matplotlib

matplotlib.use('agg', force=True)


def overlapping_visible_view_indexes_per_point(visible_view_indexes_per_point, visible_interval):
    temp_array = np.copy(visible_view_indexes_per_point)
    view_count = visible_view_indexes_per_point.shape[1]
    for i in range(view_count):
        visible_view_indexes_per_point[:, i] = \
            np.sum(temp_array[:, max(0, i - visible_interval):min(view_count, i + visible_interval)], axis=1)
    return visible_view_indexes_per_point


def get_color_file_names_by_bag(root, training_patient_id, validation_patient_id, testing_patient_id):
    training_image_list = []
    validation_image_list = []
    testing_image_list = []

    if not isinstance(training_patient_id, list):
        training_patient_id = [training_patient_id]
    if not isinstance(validation_patient_id, list):
        validation_patient_id = [validation_patient_id]
    if not isinstance(testing_patient_id, list):
        testing_patient_id = [testing_patient_id]

    for id in training_patient_id:
        training_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))
    for id in testing_patient_id:
        testing_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))
    for id in validation_patient_id:
        validation_image_list += list(root.glob('*' + str(id) + '/_start*/0*.jpg'))

    training_image_list.sort()
    testing_image_list.sort()
    validation_image_list.sort()
    return training_image_list, validation_image_list, testing_image_list


def get_color_file_names(root, split_ratio=(0.9, 0.05, 0.05)):
    image_list = list(root.glob('*/_start*/0*.jpg'))
    image_list.sort()
    split_point = [int(len(image_list) * split_ratio[0]), int(len(image_list) * (split_ratio[0] + split_ratio[1]))]
    return image_list[:split_point[0]], image_list[split_point[0]:split_point[1]], image_list[split_point[1]:]


def get_test_color_img(img_file_name, start_h, end_h, start_w, end_w, downsampling_factor):
    img = cv2.imread(img_file_name)
    downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
    downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
    downsampled_img = np.array(downsampled_img, dtype="float32")
    return downsampled_img


def get_parent_folder_names(root, id_range):
    folder_list = []
    for i in range(id_range[0], id_range[1]):
        folder_list += list(root.glob('*' + str(i) + '/_start*/'))

    folder_list.sort()
    return folder_list


def downsample_and_crop_mask(mask, downsampling_factor):
    downsampled_mask = cv2.resize(mask, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
    cropped_mask = downsampled_mask[0:128, 0:256]
    return cropped_mask, 0, 128, 0, 256


def read_selected_indexes(prefix_seq):
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))

    stride = selected_indexes[1] - selected_indexes[0]
    return stride, selected_indexes


def read_visible_image_path_list(data_root):
    visible_image_path_list = []
    visible_indexes_path_list = list(data_root.rglob("*visible_view_indexes"))
    for index_path in visible_indexes_path_list:
        with open(str(index_path)) as fp:
            for line in fp:
                visible_image_path_list.append(int(line))
    return visible_image_path_list


def read_visible_view_indexes(prefix_seq):
    visible_view_indexes = []
    with open(str(prefix_seq / 'visible_view_indexes')) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))

    return visible_view_indexes


def read_camera_intrinsic_per_view(prefix_seq):
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(str(prefix_seq / 'camera_intrinsics_per_view')) as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                param_count += 1
            elif param_count == 1:
                temp_camera_intrincis[1][1] = float(line)
                param_count += 1
            elif param_count == 2:
                temp_camera_intrincis[0][2] = float(line)
                param_count += 1
            elif param_count == 3:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0
    return camera_intrinsics


def modify_camera_intrinsic_matrix(intrinsic_matrix, downsampling_factor):
    intrinsic_matrix_modified = np.copy(intrinsic_matrix)
    intrinsic_matrix_modified[0][0] = intrinsic_matrix[0][0] / downsampling_factor
    intrinsic_matrix_modified[1][1] = intrinsic_matrix[1][1] / downsampling_factor
    intrinsic_matrix_modified[0][2] = intrinsic_matrix[0][2] / downsampling_factor
    intrinsic_matrix_modified[1][2] = intrinsic_matrix[1][2] / downsampling_factor
    return intrinsic_matrix_modified


def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path)
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])[:3]
        temp[0] = temp[0]
        temp[1] = temp[1]
        temp[2] = temp[2]
        temp.append(1.0)
        lists_3D_points.append(temp)
    return lists_3D_points


def read_view_indexes_per_point(prefix_seq, visible_view_indexes, point_cloud_count):
    # Read the view indexes per point into a 2-dimension binary matrix
    view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(str(prefix_seq / 'view_indexes_per_point')) as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
    return view_indexes_per_point


def read_pose_data(prefix_seq):
    stream = open(str(prefix_seq / "motion.yaml"), 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]
    return poses


def global_scale_estimation(extrinsics, point_cloud):
    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)

    for i, extrinsic in enumerate(extrinsics):
        if i == 0:
            max_bound = extrinsic[:3, 3]
            min_bound = extrinsic[:3, 3]
        else:
            temp = extrinsic[:3, 3]
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_1 = np.linalg.norm(max_bound - min_bound, ord=2)

    max_bound = np.zeros((3,), dtype=np.float32)
    min_bound = np.zeros((3,), dtype=np.float32)
    for i, point in enumerate(point_cloud):
        if i == 0:
            max_bound = np.asarray(point[:3], dtype=np.float32)
            min_bound = np.asarray(point[:3], dtype=np.float32)
        else:
            temp = np.asarray(point[:3], dtype=np.float32)
            if np.any(np.isnan(temp)):
                continue
            max_bound = np.maximum(max_bound, temp)
            min_bound = np.minimum(min_bound, temp)

    norm_2 = np.linalg.norm(max_bound - min_bound, ord=2)

    return max(1.0, max(norm_1, norm_2))


def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        # transform = np.linalg.inv(transform)

        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))

    return extrinsic_matrices, projection_matrices


def get_color_imgs(prefix_seq, visible_view_indexes, start_h, end_h, start_w, end_w, downsampling_factor):
    imgs = []
    for i in visible_view_indexes:
        img = cv2.imread(str(prefix_seq / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        cropped_downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        imgs.append(cropped_downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.array(imgs, dtype="float32")
    imgs = np.reshape(imgs, (-1, height, width, channel))
    return imgs


def compute_sanity_threshold(sanity_array, inlier_percentage):
    # Use histogram to cluster into different contaminated levels
    hist, bin_edges = np.histogram(sanity_array, bins=np.arange(1000) * np.max(sanity_array) / 1000.0,
                                   density=True)
    histogram_percentage = hist * np.diff(bin_edges)
    percentage = inlier_percentage
    # Let's assume there are a certain percent of points in each frame that are not contaminated
    # Get sanity threshold from counting histogram bins
    max_index = np.argmax(histogram_percentage)
    histogram_sum = histogram_percentage[max_index]
    pos_counter = 1
    neg_counter = 1
    # Assume the sanity value is a one-peak distribution
    while True:
        if max_index + pos_counter < len(histogram_percentage):
            histogram_sum = histogram_sum + histogram_percentage[max_index + pos_counter]
            pos_counter = pos_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index - neg_counter >= 0:
            histogram_sum = histogram_sum + histogram_percentage[max_index - neg_counter]
            neg_counter = neg_counter + 1
            if histogram_sum >= percentage:
                sanity_threshold_max = bin_edges[max_index + pos_counter]
                sanity_threshold_min = bin_edges[max_index - neg_counter + 1]
                break

        if max_index + pos_counter >= len(histogram_percentage) and max_index - neg_counter < 0:
            sanity_threshold_max = np.max(bin_edges)
            sanity_threshold_min = np.min(bin_edges)
            break
    return sanity_threshold_min, sanity_threshold_max


def get_clean_point_list(imgs, point_cloud, view_indexes_per_point, mask_boundary, inlier_percentage,
                         projection_matrices,
                         extrinsic_matrices):
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    if inlier_percentage <= 0.0 or inlier_percentage >= 1.0:
        return list()

    point_cloud_contamination_accumulator = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    point_cloud_appearance_count = np.zeros(array_3D_points.shape[0], dtype=np.int32)
    height, width, channel = imgs[0].shape
    valid_frame_count = 0
    mask_boundary = mask_boundary.reshape((-1, 1))
    for i in range(len(projection_matrices)):
        img = imgs[i]
        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]
        img = np.array(img, dtype=np.float32) / 255.0
        # imgs might be in HSV or BGR colorspace depending on the settings beyond this function
        img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
        img_filtered = cv2.bilateralFilter(src=img_bgr, d=7, sigmaColor=25, sigmaSpace=25)
        img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV_FULL)

        view_indexes_frame = np.asarray(view_indexes_per_point[:, i]).reshape((-1))
        visible_point_indexes = np.where(view_indexes_frame > 0.5)
        visible_point_indexes = visible_point_indexes[0]
        points_3D_camera = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
        points_3D_camera = points_3D_camera / points_3D_camera[:, 3].reshape((-1, 1))

        points_2D_image = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
        points_2D_image = points_2D_image / points_2D_image[:, 2].reshape((-1, 1))

        visible_points_2D_image = points_2D_image[visible_point_indexes, :].reshape((-1, 3))
        visible_points_3D_camera = points_3D_camera[visible_point_indexes, :].reshape((-1, 4))
        indexes = np.where((visible_points_2D_image[:, 0] <= width - 1) & (visible_points_2D_image[:, 0] >= 0) &
                           (visible_points_2D_image[:, 1] <= height - 1) & (visible_points_2D_image[:, 1] >= 0)
                           & (visible_points_3D_camera[:, 2] > 0))
        indexes = indexes[0]
        in_image_point_1D_locations = (np.round(visible_points_2D_image[indexes, 0]) +
                                       np.round(visible_points_2D_image[indexes, 1]) * width).astype(
            np.int32).reshape((-1))
        temp_mask = mask_boundary[in_image_point_1D_locations, :]
        indexes_2 = np.where(temp_mask[:, 0] == 255)
        indexes_2 = indexes_2[0]
        in_mask_point_1D_locations = in_image_point_1D_locations[indexes_2]
        points_depth = visible_points_3D_camera[indexes[indexes_2], 2]
        img_hsv = img_hsv.reshape((-1, 3))
        points_brightness = img_hsv[in_mask_point_1D_locations, 2]
        sanity_array = points_depth ** 2 * points_brightness
        point_cloud_appearance_count[visible_point_indexes[indexes[indexes_2]]] += 1
        if sanity_array.shape[0] < 2:
            continue
        valid_frame_count += 1
        sanity_threshold_min, sanity_threshold_max = compute_sanity_threshold(sanity_array, inlier_percentage)
        indexes_3 = np.where((sanity_array <= sanity_threshold_min) | (sanity_array >= sanity_threshold_max))
        indexes_3 = indexes_3[0]
        point_cloud_contamination_accumulator[visible_point_indexes[indexes[indexes_2[indexes_3]]]] += 1

    clean_point_cloud_array = (point_cloud_contamination_accumulator < point_cloud_appearance_count / 2).astype(
        np.float32)
    print("{} points eliminated".format(int(clean_point_cloud_array.shape[0] - np.sum(clean_point_cloud_array))))
    return clean_point_cloud_array


def generating_pos_and_increment(idx, visible_view_indexes, adjacent_range):
    # We use the remainder of the overall idx to retrieve the visible view
    visible_view_idx = idx % len(visible_view_indexes)

    adjacent_range_list = []
    adjacent_range_list.append(adjacent_range[0])
    adjacent_range_list.append(adjacent_range[1])

    if len(visible_view_indexes) <= 2 * adjacent_range_list[0]:
        adjacent_range_list[0] = len(visible_view_indexes) // 2

    if visible_view_idx <= adjacent_range_list[0] - 1:
        increment = random.randint(adjacent_range_list[0],
                                   min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
    elif visible_view_idx >= len(visible_view_indexes) - adjacent_range_list[0]:
        increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    else:
        # which direction should we increment
        direction = random.randint(0, 1)
        if direction == 1:
            increment = random.randint(adjacent_range_list[0],
                                       min(adjacent_range_list[1], len(visible_view_indexes) - 1 - visible_view_idx))
        else:
            increment = -random.randint(adjacent_range_list[0], min(adjacent_range_list[1], visible_view_idx))

    return [visible_view_idx, increment]


def get_pair_color_imgs(prefix_seq, pair_indexes, start_h, end_h, start_w, end_w, downsampling_factor,
                        rgb_mode):
    imgs = []
    for i in pair_indexes:
        img = cv2.imread(str(Path(prefix_seq) / "{:08d}.jpg".format(i)))
        downsampled_img = cv2.resize(img, (0, 0), fx=1. / downsampling_factor, fy=1. / downsampling_factor)
        downsampled_img = downsampled_img[start_h:end_h, start_w:end_w, :]
        if rgb_mode == "rgb":
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
        imgs.append(downsampled_img)
    height, width, channel = imgs[0].shape
    imgs = np.asarray(imgs, dtype=np.uint8)
    imgs = imgs.reshape((-1, height, width, channel))
    return imgs


def get_torch_training_data(pair_extrinsics, pair_projections, pair_indexes, point_cloud, mask_boundary,
                            view_indexes_per_point, clean_point_list, visible_view_indexes):
    height = mask_boundary.shape[0]
    width = mask_boundary.shape[1]
    pair_depth_mask_imgs = []
    pair_depth_imgs = []

    pair_flow_imgs = []
    flow_image_1 = np.zeros((height, width, 2), dtype=np.float32)
    flow_image_2 = np.zeros((height, width, 2), dtype=np.float32)

    pair_flow_mask_imgs = []
    flow_mask_image_1 = np.zeros((height, width, 1), dtype=np.float32)
    flow_mask_image_2 = np.zeros((height, width, 1), dtype=np.float32)

    # We only use inlier points
    array_3D_points = np.asarray(point_cloud).reshape((-1, 4))
    for i in range(2):
        projection_matrix = pair_projections[i]
        extrinsic_matrix = pair_extrinsics[i]

        if i == 0:
            points_2D_image_1 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_1 = np.round(points_2D_image_1 / points_2D_image_1[:, 2].reshape((-1, 1)))
            points_3D_camera_1 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_1 = points_3D_camera_1 / points_3D_camera_1[:, 3].reshape((-1, 1))
        else:
            points_2D_image_2 = np.einsum('ij,mj->mi', projection_matrix, array_3D_points)
            points_2D_image_2 = np.round(points_2D_image_2 / points_2D_image_2[:, 2].reshape((-1, 1)))
            points_3D_camera_2 = np.einsum('ij,mj->mi', extrinsic_matrix, array_3D_points)
            points_3D_camera_2 = points_3D_camera_2 / points_3D_camera_2[:, 3].reshape((-1, 1))
    # print(points_2D_image_1)
    mask_boundary = mask_boundary.reshape((-1, 1))
    flow_image_1 = flow_image_1.reshape((-1, 2))
    flow_image_2 = flow_image_2.reshape((-1, 2))
    flow_mask_image_1 = flow_mask_image_1.reshape((-1, 1))
    flow_mask_image_2 = flow_mask_image_2.reshape((-1, 1))

    points_2D_image_1 = points_2D_image_1.reshape((-1, 3))
    points_2D_image_2 = points_2D_image_2.reshape((-1, 3))
    points_3D_camera_1 = points_3D_camera_1.reshape((-1, 4))
    points_3D_camera_2 = points_3D_camera_2.reshape((-1, 4))

    point_visibility_1 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[0])]).reshape(
        (-1))
    if len(clean_point_list) != 0:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_1 = np.where((point_visibility_1 > 0.5))
    visible_point_indexes_1 = visible_point_indexes_1[0]
    point_visibility_2 = np.asarray(view_indexes_per_point[:, visible_view_indexes.index(pair_indexes[1])]).reshape(
        (-1))

    if len(clean_point_list) != 0:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5) & (clean_point_list > 0.5))
    else:
        visible_point_indexes_2 = np.where((point_visibility_2 > 0.5))
    visible_point_indexes_2 = visible_point_indexes_2[0]
    visible_points_3D_camera_1 = points_3D_camera_1[visible_point_indexes_1, :].reshape((-1, 4))
    visible_points_2D_image_1 = points_2D_image_1[visible_point_indexes_1, :].reshape((-1, 3))
    visible_points_3D_camera_2 = points_3D_camera_2[visible_point_indexes_2, :].reshape((-1, 4))
    visible_points_2D_image_2 = points_2D_image_2[visible_point_indexes_2, :].reshape((-1, 3))

    in_image_indexes_1 = np.where(
        (visible_points_2D_image_1[:, 0] <= width - 1) & (visible_points_2D_image_1[:, 0] >= 0) &
        (visible_points_2D_image_1[:, 1] <= height - 1) & (visible_points_2D_image_1[:, 1] >= 0)
        & (visible_points_3D_camera_1[:, 2] > 0))
    in_image_indexes_1 = in_image_indexes_1[0]
    in_image_point_1D_locations_1 = (np.round(visible_points_2D_image_1[in_image_indexes_1, 0]) +
                                     np.round(visible_points_2D_image_1[in_image_indexes_1, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_1 = mask_boundary[in_image_point_1D_locations_1, :]
    in_mask_indexes_1 = np.where(temp_mask_1[:, 0] == 255)
    in_mask_indexes_1 = in_mask_indexes_1[0]
    in_mask_point_1D_locations_1 = in_image_point_1D_locations_1[in_mask_indexes_1]
    flow_mask_image_1[in_mask_point_1D_locations_1, 0] = 1.0

    in_image_indexes_2 = np.where(
        (visible_points_2D_image_2[:, 0] <= width - 1) & (visible_points_2D_image_2[:, 0] >= 0) &
        (visible_points_2D_image_2[:, 1] <= height - 1) & (visible_points_2D_image_2[:, 1] >= 0)
        & (visible_points_3D_camera_2[:, 2] > 0))
    in_image_indexes_2 = in_image_indexes_2[0]
    in_image_point_1D_locations_2 = (np.round(visible_points_2D_image_2[in_image_indexes_2, 0]) +
                                     np.round(visible_points_2D_image_2[in_image_indexes_2, 1]) * width).astype(
        np.int32).reshape((-1))
    temp_mask_2 = mask_boundary[in_image_point_1D_locations_2, :]
    in_mask_indexes_2 = np.where(temp_mask_2[:, 0] == 255)
    in_mask_indexes_2 = in_mask_indexes_2[0]
    in_mask_point_1D_locations_2 = in_image_point_1D_locations_2[in_mask_indexes_2]
    flow_mask_image_2[in_mask_point_1D_locations_2, 0] = 1.0

    flow_image_1[in_mask_point_1D_locations_1, :] = points_2D_image_2[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]],
                                                    :2] - \
                                                    points_2D_image_1[
                                                    visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], :2]
    flow_image_2[in_mask_point_1D_locations_2, :] = points_2D_image_1[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]],
                                                    :2] - \
                                                    points_2D_image_2[
                                                    visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], :2]

    flow_image_1[:, 0] /= width
    flow_image_1[:, 1] /= height
    flow_image_2[:, 0] /= width
    flow_image_2[:, 1] /= height

    outlier_indexes_1 = np.where((np.abs(flow_image_1[:, 0]) > 5.0) | (np.abs(flow_image_1[:, 1]) > 5.0))[0]
    outlier_indexes_2 = np.where((np.abs(flow_image_2[:, 0]) > 5.0) | (np.abs(flow_image_2[:, 1]) > 5.0))[0]
    flow_mask_image_1[outlier_indexes_1, 0] = 0.0
    flow_mask_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 0] = 0.0
    flow_image_2[outlier_indexes_2, 0] = 0.0
    flow_image_1[outlier_indexes_1, 1] = 0.0
    flow_image_2[outlier_indexes_2, 1] = 0.0

    depth_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_1 = np.zeros((height, width, 1), dtype=np.float32)
    depth_mask_img_2 = np.zeros((height, width, 1), dtype=np.float32)
    depth_img_1 = depth_img_1.reshape((-1, 1))
    depth_img_2 = depth_img_2.reshape((-1, 1))
    depth_mask_img_1 = depth_mask_img_1.reshape((-1, 1))
    depth_mask_img_2 = depth_mask_img_2.reshape((-1, 1))

    depth_img_1[in_mask_point_1D_locations_1, 0] = points_3D_camera_1[
        visible_point_indexes_1[in_image_indexes_1[in_mask_indexes_1]], 2]
    depth_img_2[in_mask_point_1D_locations_2, 0] = points_3D_camera_2[
        visible_point_indexes_2[in_image_indexes_2[in_mask_indexes_2]], 2]
    depth_mask_img_1[in_mask_point_1D_locations_1, 0] = 1.0
    depth_mask_img_2[in_mask_point_1D_locations_2, 0] = 1.0

    pair_flow_imgs.append(flow_image_1)
    pair_flow_imgs.append(flow_image_2)
    pair_flow_imgs = np.array(pair_flow_imgs, dtype="float32")
    pair_flow_imgs = np.reshape(pair_flow_imgs, (-1, height, width, 2))

    pair_flow_mask_imgs.append(flow_mask_image_1)
    pair_flow_mask_imgs.append(flow_mask_image_2)
    pair_flow_mask_imgs = np.array(pair_flow_mask_imgs, dtype="float32")
    pair_flow_mask_imgs = np.reshape(pair_flow_mask_imgs, (-1, height, width, 1))

    pair_depth_mask_imgs.append(depth_mask_img_1)
    pair_depth_mask_imgs.append(depth_mask_img_2)
    pair_depth_mask_imgs = np.array(pair_depth_mask_imgs, dtype="float32")
    pair_depth_mask_imgs = np.reshape(pair_depth_mask_imgs, (-1, height, width, 1))

    pair_depth_imgs.append(depth_img_1)
    pair_depth_imgs.append(depth_img_2)
    pair_depth_imgs = np.array(pair_depth_imgs, dtype="float32")
    pair_depth_imgs = np.reshape(pair_depth_imgs, (-1, height, width, 1))

    return pair_depth_mask_imgs, pair_depth_imgs, pair_flow_mask_imgs, pair_flow_imgs


def save_model(model, optimizer, epoch, step, model_path, validation_loss):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'validation': validation_loss
    }, str(model_path))
    return


def display_depth_map(depth_map, min_value=None, max_value=None, colormode=cv2.COLORMAP_JET):
    if min_value is None or max_value is None:
        min_value = np.min(depth_map)
        max_value = np.max(depth_map)
    depth_map_visualize = np.abs((depth_map - min_value) / (max_value - min_value) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), colormode)
    return depth_map_visualize


def point_cloud_from_depth(depth_map, color_img, mask_img, intrinsic_matrix, point_cloud_downsampling,
                           min_threshold=None, max_threshold=None):
    point_clouds = []
    height, width, channel = color_img.shape

    f_x = intrinsic_matrix[0, 0]
    c_x = intrinsic_matrix[0, 2]
    f_y = intrinsic_matrix[1, 1]
    c_y = intrinsic_matrix[1, 2]

    for h in range(height):
        for w in range(width):
            if h % point_cloud_downsampling == 0 and w % point_cloud_downsampling == 0 and mask_img[h, w] > 0.5:
                z = depth_map[h, w]
                x = (w - c_x) / f_x * z
                y = (h - c_y) / f_y * z
                b = color_img[h, w, 0]
                g = color_img[h, w, 1]
                r = color_img[h, w, 2]
                if max_threshold is not None and min_threshold is not None:
                    if np.max([r, g, b]) >= max_threshold and np.min([r, g, b]) <= min_threshold:
                        point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))
                else:
                    point_clouds.append((x, y, z, np.uint8(r), np.uint8(g), np.uint8(b)))

    point_clouds = np.array(point_clouds, dtype='float32')
    point_clouds = np.reshape(point_clouds, (-1, 6))
    return point_clouds


def write_point_cloud(path, point_cloud):
    point_clouds_list = []
    for i in range(point_cloud.shape[0]):
        point_clouds_list.append((point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2], point_cloud[i, 3],
                                  point_cloud[i, 4], point_cloud[i, 5]))

    vertex = np.array(point_clouds_list,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(path)
    return


def draw_flow(flows, max_v=None):
    batch_size, channel, height, width = flows.shape
    flows_x_display = vutils.make_grid(flows[:, 0, :, :].reshape(batch_size, 1, height, width), normalize=False,
                                       scale_each=False)
    flows_y_display = vutils.make_grid(flows[:, 1, :, :].reshape(batch_size, 1, height, width), normalize=False,
                                       scale_each=False)
    flows_display = torch.cat([flows_x_display[0, :, :].reshape(1, flows_x_display.shape[1], flows_x_display.shape[2]),
                               flows_y_display[0, :, :].reshape(1, flows_x_display.shape[1], flows_x_display.shape[2])],
                              dim=0)
    flows_display = flows_display.data.cpu().numpy()
    flows_display = np.moveaxis(flows_display, source=[0, 1, 2], destination=[2, 0, 1])
    h, w = flows_display.shape[:2]
    fx, fy = flows_display[:, :, 0], flows_display[:, :, 1] * h / w
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    if max_v is None:
        hsv[..., 2] = np.uint8(np.minimum(v / np.max(v), 1.0) * 255)
    else:
        hsv[..., 2] = np.uint8(np.minimum(v / max_v, 1.0) * 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.max(v)


def stack_and_display(phase, title, step, writer, image_list, return_image=False):
    writer.add_image(phase + '/Images/' + title,
                     np.moveaxis(np.vstack(image_list), source=[0, 1, 2], destination=[1, 2, 0]), step)
    if return_image:
        return np.vstack(image_list)
    else:
        return


def display_color_sparse_depth_dense_depth_warped_depth_sparse_flow_dense_flow(idx, step, writer, colors_1,
                                                                               sparse_depths_1, pred_depths_1,
                                                                               warped_depths_2_to_1,
                                                                               sparse_flows_1, flows_from_depth_1,
                                                                               boundaries,
                                                                               phase="Training", is_return_image=False,
                                                                               color_reverse=True,
                                                                               rgb_mode="bgr",
                                                                               ):
    colors_display = vutils.make_grid((colors_1 * 0.5 + 0.5) * boundaries, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])

    if rgb_mode == "bgr":
        colors_display = cv2.cvtColor(colors_display, cv2.COLOR_BGR2RGB)

    min_depth = torch.min(pred_depths_1)
    max_depth = torch.max(pred_depths_1)

    pred_depths_display = vutils.make_grid(pred_depths_1, normalize=True, scale_each=False,
                                           range=(min_depth.item(), max_depth.item()))
    pred_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depths_display.data.cpu().numpy(),
                                                                       source=[0, 1, 2],
                                                                       destination=[2, 0, 1])), cv2.COLORMAP_JET)

    sparse_depths_display = vutils.make_grid(sparse_depths_1, normalize=True, scale_each=False,
                                             range=(min_depth.item(), max_depth.item()))
    sparse_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(sparse_depths_display.data.cpu().numpy(),
                                                                         source=[0, 1, 2],
                                                                         destination=[2, 0, 1])), cv2.COLORMAP_JET)

    warped_depths_display = vutils.make_grid(warped_depths_2_to_1, normalize=True, scale_each=False,
                                             range=(min_depth.item(), max_depth.item()))
    warped_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(warped_depths_display.data.cpu().numpy(),
                                                                         source=[0, 1, 2],
                                                                         destination=[2, 0, 1])), cv2.COLORMAP_JET)

    dense_flows_display, max_v = draw_flow(flows_from_depth_1)
    sparse_flows_display, _ = draw_flow(sparse_flows_1, max_v=max_v)

    if color_reverse:
        pred_depths_display = cv2.cvtColor(pred_depths_display, cv2.COLOR_BGR2RGB)
        warped_depths_display = cv2.cvtColor(warped_depths_display, cv2.COLOR_BGR2RGB)
        sparse_depths_display = cv2.cvtColor(sparse_depths_display, cv2.COLOR_BGR2RGB)
        dense_flows_display = cv2.cvtColor(dense_flows_display, cv2.COLOR_BGR2RGB)
        sparse_flows_display = cv2.cvtColor(sparse_flows_display, cv2.COLOR_BGR2RGB)
    if is_return_image:
        return colors_display, sparse_depths_display.astype(np.float32) / 255.0, pred_depths_display.astype(
            np.float32) / 255.0, warped_depths_display.astype(np.float32) / 255.0, sparse_flows_display.astype(
            np.float32) / 255.0, dense_flows_display.astype(np.float32) / 255.0
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Sparse_Depth_' + str(idx), sparse_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Pred_Depth_' + str(idx), pred_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Warped_Depth_' + str(idx), warped_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Sparse_Flow_' + str(idx), sparse_flows_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Dense_Flow_' + str(idx), dense_flows_display, step, dataformats="HWC")
        return


@torch.no_grad()
def display_color_depth_sparse_flow_dense_flow(idx, step, writer, colors_1, pred_depths_1,
                                               sparse_flows_1, flows_from_depth_1,
                                               phase="Training", is_return_image=False, color_reverse=True
                                               ):
    colors_display = vutils.make_grid(colors_1 * 0.5 + 0.5, normalize=False)
    colors_display = np.moveaxis(colors_display.data.cpu().numpy(),
                                 source=[0, 1, 2], destination=[2, 0, 1])

    pred_depths_display = vutils.make_grid(pred_depths_1, normalize=True, scale_each=True)
    pred_depths_display = cv2.applyColorMap(np.uint8(255 * np.moveaxis(pred_depths_display.data.cpu().numpy(),
                                                                       source=[0, 1, 2],
                                                                       destination=[2, 0, 1])), cv2.COLORMAP_JET)
    sparse_flows_display, max_v = draw_flow(sparse_flows_1)
    dense_flows_display, _ = draw_flow(flows_from_depth_1, max_v=max_v)
    if color_reverse:
        pred_depths_display = cv2.cvtColor(pred_depths_display, cv2.COLOR_BGR2RGB)
        sparse_flows_display = cv2.cvtColor(sparse_flows_display, cv2.COLOR_BGR2RGB)
        dense_flows_display = cv2.cvtColor(dense_flows_display, cv2.COLOR_BGR2RGB)

    if is_return_image:
        return colors_display, pred_depths_display.astype(np.float32) / 255.0, \
               sparse_flows_display.astype(np.float32) / 255.0, dense_flows_display.astype(np.float32) / 255.0
    else:
        writer.add_image(phase + '/Images/Color_' + str(idx), colors_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Pred_Depth_' + str(idx), pred_depths_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Sparse_Flow_' + str(idx), sparse_flows_display, step, dataformats="HWC")
        writer.add_image(phase + '/Images/Dense_Flow_' + str(idx), dense_flows_display, step, dataformats="HWC")
        return


def get_filenames_from_frame_indexes(sequence_root, frame_index_array):
    test_image_list = []
    for index in frame_index_array:
        temp = list(sequence_root.rglob('{:08d}.jpg'.format(index)))
        if len(temp) != 0:
            test_image_list.append(temp[0])
    test_image_list.sort()
    return test_image_list


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


