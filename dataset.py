import os
from skimage import io

from torch.utils.data import Dataset


class SkalpDataset(Dataset):
    def __init__(self, data_dir, file, transform=None):
        self.data_dir = data_dir
        with open(file) as f:
            self.all_images = [i.strip().zfill(8) + ".jpg" for i in f]
        self.right_images = self.all_images[::2]
        self.left_images = self.all_images[1::2]
        self.transform = transform

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, item):
        image_right = io.imread(os.path.join(self.data_dir, self.right_images[item]))
        image_left = io.imread(os.path.join(self.data_dir, self.left_images[item]))

        if self.transform:
            image_right = self.transform(image_right)
            image_left = self.transform(image_left)

        return image_right, image_left


# f = open(r"D:\Programowanie\DL\Inzynierka\DepthEstimation\training_data\bag_1\_start_saki4\selected_indexes", "r")
# print(f.read())

# dataset = SkalpDataset(r"training_data/bag_1/_start_saki4", r"training_data/bag_1/_start_saki4/selected_indexes",
#                        ToTensor())
# first_data = dataset[0]
# # _, ax = plt.subplots(1, 2)
# # ax[0].imshow(first_data[0])
# # ax[1].imshow(first_data[1])
# # plt.show()
# print(first_data[0].shape, first_data[1].shape)
