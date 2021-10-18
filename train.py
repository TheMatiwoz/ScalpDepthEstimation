import argparse
from datetime import datetime
from pathlib import Path
import albumentations as albu

import cv2

from tools import utils


if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Self-supervised Depth Estimation on Monocular Endoscopy Dataset -- Train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adjacent_range', nargs='+', type=int, required=True,
                        help='interval range for a pair of video frames')
    parser.add_argument('--id_range', nargs='+', type=int, required=True,
                        help='id range for the training and testing dataset')
    parser.add_argument('--input_downsampling', type=float, default=4.0,
                        help='image downsampling rate')
    parser.add_argument('--input_size', nargs='+', type=int, required=True, help='resolution of network input')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for input data loader')
    parser.add_argument('--num_pre_workers', type=int, default=8,
                        help='number of workers for preprocessing intermediate data')
    parser.add_argument('--dcl_weight', type=float, default=5.0,
                        help='weight for depth consistency loss in the later training stage')
    parser.add_argument('--sfl_weight', type=float, default=20.0, help='weight for sparse flow loss')
    parser.add_argument('--max_lr', type=float, default=1.0e-3, help='upper bound learning rate for cyclic lr')
    parser.add_argument('--min_lr', type=float, default=1.0e-4, help='lower bound learning rate for cyclic lr')
    parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations per epoch')
    parser.add_argument('--network_downsampling', type=int, default=64, help='network downsampling of the input image')
    parser.add_argument('--inlier_percentage', type=float, default=0.99,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--validation_interval', type=int, default=1, help='epoch interval for validation')
    parser.add_argument('--zero_division_epsilon', type=float, default=1.0e-8, help='epsilon to prevent zero division')
    parser.add_argument('--display_interval', type=int, default=10, help='iteration interval for image display')
    parser.add_argument('--training_patient_id', nargs='+', type=int, required=True, help='id of the training patient')
    parser.add_argument('--testing_patient_id', nargs='+', type=int, required=True, help='id of the testing patient')
    parser.add_argument('--validation_patient_id', nargs='+', type=int, required=True,
                        help='id of the valiadtion patient')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--load_trained_model', action='store_true',
                        help='whether to load trained student model')
    parser.add_argument('--number_epoch', type=int, required=True, help='number of epochs in total')
    parser.add_argument('--visibility_overlap', type=int, default=30, help='overlap of point visibility information')
    parser.add_argument('--use_hsv_colorspace', action='store_true',
                        help='convert RGB to hsv colorspace')
    parser.add_argument('--training_result_root', type=str, required=True, help='root of the training input and ouput')
    parser.add_argument('--training_data_root', type=str, required=True, help='path to the training data')
    parser.add_argument('--architecture_summary', action='store_true', help='display the network architecture')
    parser.add_argument('--trained_model_path', type=str, default=None,
                        help='path to the trained student model')

    args = parser.parse_args()

# Hyper-parameters
    adjacent_range = args.adjacent_range
    input_downsampling = args.input_downsampling
    height, width = args.input_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_pre_workers = args.num_pre_workers
    depth_consistency_weight = args.dcl_weight
    sparse_flow_weight = args.sfl_weight
    max_lr = args.max_lr
    min_lr = args.min_lr
    num_iter = args.num_iter
    network_downsampling = args.network_downsampling
    inlier_percentage = args.inlier_percentage
    validation_each = args.validation_interval
    depth_scaling_epsilon = args.zero_division_epsilon
    depth_warping_epsilon = args.zero_division_epsilon
    wsl_epsilon = args.zero_division_epsilon
    display_each = args.display_interval
    training_patient_id = args.training_patient_id
    testing_patient_id = args.testing_patient_id
    validation_patient_id = args.validation_patient_id
    load_intermediate_data = args.load_intermediate_data
    load_trained_model = args.load_trained_model
    n_epochs = args.number_epoch
    is_hsv = args.use_hsv_colorspace
    training_result_root = args.training_result_root
    display_architecture = args.architecture_summary
    trained_model_path = args.trained_model_path
    training_data_root = Path(args.training_data_root)
    id_range = args.id_range
    visibility_overlap = args.visibility_overlap
    currentDT = datetime.now()

    training_transforms = albu.Compose([
        # Color augmentation
        albu.OneOf([
            albu.Compose([
                albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                albu.RandomGamma(gamma_limit=(80, 120), p=0.5),
                albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=0, val_shift_limit=0, p=0.5)]),
            albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5)
        ]),
        # Image quality augmentation
        albu.OneOf([
            albu.Blur(p=0.5),
            albu.MedianBlur(p=0.5),
            albu.MotionBlur(p=0.5),
            albu.JpegCompression(quality_lower=20, quality_upper=100, p=0.5)
        ]),
        # Noise augmentation
        albu.OneOf([
            albu.GaussNoise(var_limit=(10, 30), p=0.5),
            albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.005 * 255, 0.02 * 255), p=0.5)
        ]),
    ], p=1.)

    log_root = Path(training_result_root) / "depth_estimation_train_run_{}_{}_{}_{}_test_id_{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour,
        currentDT.minute,
        "_".join(str(testing_patient_id)))

    # Get color image filenames
    train_filenames, val_filenames, test_filenames = utils.get_color_file_names_by_bag(training_data_root,
                                                                                       training_patient_id=training_patient_id,
                                                                                       validation_patient_id=validation_patient_id,
                                                                                       testing_patient_id=testing_patient_id)

    folder_list = utils.get_parent_folder_names(training_data_root, id_range=id_range)
