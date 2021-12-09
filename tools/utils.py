import numpy as np
import math
import matplotlib.pyplot as plt

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

    for train_id in training_patient_id:
        training_image_list += list(root.glob('*' + str(train_id) + '/_start*/0*.jpg'))
    for test_id in testing_patient_id:
        testing_image_list += list(root.glob('*' + str(test_id) + '/_start*/0*.jpg'))
    for val_id in validation_patient_id:
        validation_image_list += list(root.glob('*' + str(val_id) + '/_start*/0*.jpg'))

    training_image_list.sort()
    testing_image_list.sort()
    validation_image_list.sort()
    return training_image_list, validation_image_list, testing_image_list


def get_parent_folder_names(root, id_range):
    folder_list = []
    for i in range(id_range[0], id_range[1]):
        folder_list += list(root.glob('*' + str(i) + '/_start*/'))

    folder_list.sort()
    return folder_list


def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


# print(distance(1.69877,1.10727,1.69439,1.10135))
# print(distance(-9.13142, -3.99888, -8.26979, -3.34355))
# img = plt.imread(r"D:\Programowanie\DL\Inzynierka\DepthEstimation\training_data\bag_1\_start_saki4\00000001.jpg")
# plt.imshow(img)
# plt.show()

# ar = np.ones((3,4))
# print(ar)
# print(-ar)

