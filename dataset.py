from torch.utils.data import Dataset
import albumentations as albu


class SfMDataset(Dataset):
    def __init__(self, image_file_names, folder_list, adjacent_range,
                 transform, downsampling, network_downsampling, inlier_percentage, visible_interval,
                 use_store_data, store_data_root, phase, rgb_mode, num_iter):
        self.image_file_names = image_file_names
        self.folder_list = folder_list
        self.adjacent_range = adjacent_range
        self.transform = transform
        self.downsampling = downsampling
        self.network_downsampling = network_downsampling
        self.inlier_percentage = inlier_percentage
        self.visible_interval = visible_interval
        self.use_store_data = use_store_data
        self.store_data_root = store_data_root
        self.phase = phase
        self.rgb_mode = rgb_mode
        self.num_iter = num_iter
        self.n_samples = len(self.image_file_names)

        self.clean_point_list_per_seq = {}
        self.intrinsic_matrix_per_seq = {}
        self.point_cloud_per_seq = {}
        self.mask_boundary_per_seq = {}
        self.view_indexes_per_point_per_seq = {}
        self.selected_indexes_per_seq = {}
        self.visible_view_indexes_per_seq = {}
        self.extrinsics_per_seq = {}
        self.projection_per_seq = {}
        self.crop_positions_per_seq = {}
        self.estimated_scale_per_seq = {}
        self.normalize = albu.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5), max_pixel_value=255.0)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
