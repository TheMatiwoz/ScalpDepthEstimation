from pathlib import Path


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

