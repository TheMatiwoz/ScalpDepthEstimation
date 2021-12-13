import tqdm
import numpy as np
from pathlib import Path
import torchsummary
import math
import torch
import random
from tensorboardX import SummaryWriter

import argparse
import datetime
# Local

import layers
import models
import losses
from tools import utils
import dataset
import scheduler


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Scalp Depth Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training_result_root', type=str, required=True, help='root of the training and output')
    parser.add_argument('--training_data_root', type=str, required=True, help='path to the training data')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)

    adjacent_range = [5, 15]
    input_downsampling = 5
    height, width = 128, 256
    batch_size = 8
    dc_weight = 5
    sf_weight = 20
    max_lr = 1.0e-3
    min_lr = 1.0e-4
    num_iter = 800
    inlier_percentage = 0.99
    epsilon = 1.0e-8
    display_each = 20
    training_patient_id = [3, 5]
    testing_patient_id = 2
    validation_patient_id = 4
    n_epochs = 50
    training_result_root = args.training_result_root
    training_data_root = Path(args.training_data_root)
    id_range = [2, 6]
    visibility_overlap = 10
    currentDT = datetime.datetime.now()

    log_root = Path(training_result_root) / "depth_estimation_train_run_{}_{}_{}_{}_test_id_{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour,
        currentDT.minute,
        "_".join(str(testing_patient_id)))
    if not log_root.exists():
        log_root.mkdir()
    writer = SummaryWriter(logdir=str(log_root))
    print("Tensorboard visualization at {}".format(str(log_root)))

    train_filenames, val_filenames, test_filenames = utils.get_color_file_names_by_bag(training_data_root,
                                                                                       training_patient_id=training_patient_id,
                                                                                       validation_patient_id=validation_patient_id,
                                                                                       testing_patient_id=testing_patient_id)
    folder_list = utils.get_parent_folder_names(training_data_root, id_range=id_range)

    train_dataset = dataset.SfMDataset(image_file_names=train_filenames, folder_list=folder_list,
                                       adjacent_range=adjacent_range, downsampling=input_downsampling,
                                       inlier_percentage=inlier_percentage, visible_interval=visibility_overlap,
                                       store_data_root=training_data_root, phase="train", rgb_mode="rgb",
                                       num_iter=num_iter)
    validation_dataset = dataset.SfMDataset(image_file_names=val_filenames, folder_list=folder_list,
                                            adjacent_range=adjacent_range, downsampling=input_downsampling,
                                            inlier_percentage=inlier_percentage, visible_interval=visibility_overlap,
                                            store_data_root=training_data_root, phase="validation", rgb_mode="rgb",
                                            num_iter=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=batch_size)

    depth_estimation_model = models.FCDenseNet().cuda()
    torchsummary.summary(depth_estimation_model, input_size=(3, height, width))

    optimizer = torch.optim.Adam(depth_estimation_model.parameters(), lr=min_lr)
    lr_scheduler = scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=num_iter)

    depth_scaling_layer = layers.DepthScalingLayer(epsilon=epsilon)
    depth_warping_layer = layers.DepthWarpingLayer(epsilon=epsilon)
    flow_from_depth_layer = layers.FlowfromDepthLayer()

    sparse_flow_loss_function = losses.SparseMaskedL1Loss()
    depth_consistency_loss_function = losses.NormalizedDistanceLoss(height=height, width=width)

    epoch = 0
    step = 0
    flag = 0

    for epoch in range(epoch, n_epochs):

        # Update progress bar
        tq = tqdm.tqdm(total=len(train_loader) * batch_size, dynamic_ncols=True, ncols=40)
        # Variable initialization
        if epoch <= 20:
            depth_consistency_weight = 0.1
        else:
            depth_consistency_weight = dc_weight
        for batch, (
                colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2,
                sparse_flows_1, sparse_flows_2, sparse_flow_masks_1, sparse_flow_masks_2, boundaries, rotations_1_wrt_2,
                rotations_2_wrt_1, translations_1_wrt_2, translations_2_wrt_1, intrinsics, folders, file_names) in \
                enumerate(train_loader):
            # Update learning rate
            lr_scheduler.batch_step(batch_iteration=step)
            tq.set_description('Epoch {}, lr {}'.format(epoch, lr_scheduler.get_lr()))

            with torch.no_grad():
                colors_1 = colors_1.cuda()
                colors_2 = colors_2.cuda()
                sparse_depths_1 = sparse_depths_1.cuda()
                sparse_depths_2 = sparse_depths_2.cuda()
                sparse_depth_masks_1 = sparse_depth_masks_1.cuda()
                sparse_depth_masks_2 = sparse_depth_masks_2.cuda()
                sparse_flows_1 = sparse_flows_1.cuda()
                sparse_flows_2 = sparse_flows_2.cuda()
                sparse_flow_masks_1 = sparse_flow_masks_1.cuda()
                sparse_flow_masks_2 = sparse_flow_masks_2.cuda()
                boundaries = boundaries.cuda()
                rotations_1_wrt_2 = rotations_1_wrt_2.cuda()
                rotations_2_wrt_1 = rotations_2_wrt_1.cuda()
                translations_1_wrt_2 = translations_1_wrt_2.cuda()
                translations_2_wrt_1 = translations_2_wrt_1.cuda()
                intrinsics = intrinsics.cuda()

            colors_1 = boundaries * colors_1
            colors_2 = boundaries * colors_2

            predicted_depth_maps_1 = depth_estimation_model(colors_1)
            predicted_depth_maps_2 = depth_estimation_model(colors_2)

            scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
                [predicted_depth_maps_1, sparse_depths_1, sparse_depth_masks_1])
            scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
                [predicted_depth_maps_2, sparse_depths_2, sparse_depth_masks_2])

            flows_from_depth_1 = flow_from_depth_layer(
                [scaled_depth_maps_1, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                 intrinsics])
            flows_from_depth_2 = flow_from_depth_layer(
                [scaled_depth_maps_2, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                 intrinsics])
            sparse_flow_masks_1 = sparse_flow_masks_1 * boundaries
            sparse_flow_masks_2 = sparse_flow_masks_2 * boundaries
            sparse_flows_1 = sparse_flows_1 * boundaries
            sparse_flows_2 = sparse_flows_2 * boundaries
            flows_from_depth_1 = flows_from_depth_1 * boundaries
            flows_from_depth_2 = flows_from_depth_2 * boundaries

            sparse_flow_loss = sf_weight * 0.5 * (sparse_flow_loss_function(
                [sparse_flows_1, flows_from_depth_1, sparse_flow_masks_1]) + sparse_flow_loss_function(
                [sparse_flows_2, flows_from_depth_2, sparse_flow_masks_2]))

            # Depth consistency loss
            warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                 intrinsics])
            warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                 intrinsics])
            depth_consistency_loss = depth_consistency_weight * 0.5 * (depth_consistency_loss_function(
                [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1,
                 intrinsics]) + depth_consistency_loss_function(
                [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, intrinsics]))
            loss = depth_consistency_loss + sparse_flow_loss

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                optimizer.zero_grad()
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                continue
            else:
                flag += 1
                optimizer.zero_grad()
                loss.backward()
                # Prevent one sample from having too much impact on the training
                torch.nn.utils.clip_grad_norm_(depth_estimation_model.parameters(), 10.0)
                optimizer.step()
                if batch == 0 or flag == 1:
                    mean_loss = loss.item()
                    mean_depth_consistency_loss = depth_consistency_loss.item()
                    mean_sparse_flow_loss = sparse_flow_loss.item()
                else:
                    mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
                    mean_depth_consistency_loss = (mean_depth_consistency_loss * batch +
                                                   depth_consistency_loss.item()) / (batch + 1.0)
                    mean_sparse_flow_loss = (mean_sparse_flow_loss * batch + sparse_flow_loss.item()) / (batch + 1.0)

            step += 1
            tq.update(batch_size)
            tq.set_postfix(loss='avg: {:.5f} cur: {:.5f}'.format(mean_loss, loss.item()),
                           loss_depth_consistency='avg: {:.5f} cur: {:.5f}'.format(
                               mean_depth_consistency_loss,
                               depth_consistency_loss.item()),
                           loss_sparse_flow='avg: {:.5f} cur: {:.5f}'.format(
                               mean_sparse_flow_loss,
                               sparse_flow_loss.item()))
            writer.add_scalars('Training', {'overall': mean_loss,
                                            'depth_consistency': mean_depth_consistency_loss,
                                            'sparse_flow': mean_sparse_flow_loss}, step)

            if batch % display_each == 0:
                colors_1_display, sparse_depths_1_display, pred_depths_1_display, warped_depths_1_display, sparse_flows_1_display, dense_flows_1_display = \
                    utils.display_color_sparse_depth_dense_depth_warped_depth_sparse_flow_dense_flow(idx=1, step=step,
                                                                                                     writer=writer,
                                                                                                     colors_1=colors_1,
                                                                                                     sparse_depths_1=sparse_depths_1,
                                                                                                     pred_depths_1=scaled_depth_maps_1 * boundaries,
                                                                                                     warped_depths_2_to_1=warped_depth_maps_2_to_1,
                                                                                                     sparse_flows_1=sparse_flows_1,
                                                                                                     flows_from_depth_1=flows_from_depth_1,
                                                                                                     boundaries=boundaries,
                                                                                                     phase="Training",
                                                                                                     is_return_image=True,
                                                                                                     color_reverse=True,
                                                                                                     rgb_mode="rgb")
                colors_2_display, sparse_depths_2_display, pred_depths_2_display, warped_depths_2_display, sparse_flows_2_display, dense_flows_2_display = \
                    utils.display_color_sparse_depth_dense_depth_warped_depth_sparse_flow_dense_flow(idx=2, step=step,
                                                                                                     writer=writer,
                                                                                                     colors_1=colors_2,
                                                                                                     sparse_depths_1=sparse_depths_2,
                                                                                                     pred_depths_1=scaled_depth_maps_2 * boundaries,
                                                                                                     warped_depths_2_to_1=warped_depth_maps_1_to_2,
                                                                                                     sparse_flows_1=sparse_flows_2,
                                                                                                     flows_from_depth_1=flows_from_depth_2,
                                                                                                     boundaries=boundaries,
                                                                                                     phase="Training",
                                                                                                     is_return_image=True,
                                                                                                     color_reverse=True,
                                                                                                     rgb_mode="rgb")
                image_display = utils.stack_and_display(phase="Training",
                                                        title="Results (c1, sd1, d1, wd1, sf1, df1, c2, sd2, d2, wd2, sf2, df2)",
                                                        step=step, writer=writer,
                                                        image_list=[colors_1_display, sparse_depths_1_display,
                                                                    pred_depths_1_display,
                                                                    warped_depths_1_display, sparse_flows_1_display,
                                                                    dense_flows_1_display,
                                                                    colors_2_display, sparse_depths_2_display,
                                                                    pred_depths_2_display,
                                                                    warped_depths_2_display, sparse_flows_2_display,
                                                                    dense_flows_2_display],
                                                        return_image=True)
            tq.close()

            tq = tqdm.tqdm(total=len(validation_loader) * batch_size, dynamic_ncols=True, ncols=40)
            tq.set_description('Validation Epoch {}'.format(epoch))
            with torch.no_grad():
                for batch, (
                        colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1,
                        sparse_depth_masks_2, sparse_flows_1,
                        sparse_flows_2, sparse_flow_masks_1, sparse_flow_masks_2, boundaries, rotations_1_wrt_2,
                        rotations_2_wrt_1, translations_1_wrt_2, translations_2_wrt_1, intrinsics,
                        folders, file_names) in enumerate(validation_loader):

                    colors_1 = colors_1.cuda()
                    colors_2 = colors_2.cuda()
                    sparse_depths_1 = sparse_depths_1.cuda()
                    sparse_depths_2 = sparse_depths_2.cuda()
                    sparse_depth_masks_1 = sparse_depth_masks_1.cuda()
                    sparse_depth_masks_2 = sparse_depth_masks_2.cuda()
                    sparse_flows_1 = sparse_flows_1.cuda()
                    sparse_flows_2 = sparse_flows_2.cuda()
                    sparse_flow_masks_1 = sparse_flow_masks_1.cuda()
                    sparse_flow_masks_2 = sparse_flow_masks_2.cuda()
                    boundaries = boundaries.cuda()
                    rotations_1_wrt_2 = rotations_1_wrt_2.cuda()
                    rotations_2_wrt_1 = rotations_2_wrt_1.cuda()
                    translations_1_wrt_2 = translations_1_wrt_2.cuda()
                    translations_2_wrt_1 = translations_2_wrt_1.cuda()
                    intrinsics = intrinsics.cuda()

                    colors_1 = boundaries * colors_1
                    colors_2 = boundaries * colors_2

                    # Predicted depth from student model
                    predicted_depth_maps_1 = depth_estimation_model(colors_1)
                    predicted_depth_maps_2 = depth_estimation_model(colors_2)
                    # predicted_depth_maps_1 = torch.nn.functional.interpolate(predicted_depth_maps_1, size=(180, 320),
                    #                                                          mode='bilinear')
                    # predicted_depth_maps_2 = torch.nn.functional.interpolate(predicted_depth_maps_2, size=(180, 320),
                    #                                                          mode='bilinear')
                    scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
                        [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
                    scaled_depth_maps_2, normalized_scale_std_2 = depth_scaling_layer(
                        [torch.abs(predicted_depth_maps_2), sparse_depths_2, sparse_depth_masks_2])

                    # Sparse flow loss
                    flows_from_depth_1 = flow_from_depth_layer(
                        [scaled_depth_maps_1, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                         intrinsics])
                    flows_from_depth_2 = flow_from_depth_layer(
                        [scaled_depth_maps_2, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                         intrinsics])
                    sparse_flow_masks_1 = sparse_flow_masks_1 * boundaries
                    sparse_flow_masks_2 = sparse_flow_masks_2 * boundaries
                    sparse_flows_1 = sparse_flows_1 * boundaries
                    sparse_flows_2 = sparse_flows_2 * boundaries
                    flows_from_depth_1 = flows_from_depth_1 * boundaries
                    flows_from_depth_2 = flows_from_depth_2 * boundaries
                    sparse_flow_loss = sf_weight * 0.5 * (sparse_flow_loss_function(
                        [sparse_flows_1, flows_from_depth_1, sparse_flow_masks_1]) + sparse_flow_loss_function(
                        [sparse_flows_2, flows_from_depth_2, sparse_flow_masks_2]))

                    # Depth consistency loss
                    warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                        [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                         intrinsics])
                    warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                        [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                         intrinsics])
                    depth_consistency_loss = depth_consistency_weight * 0.5 * (depth_consistency_loss_function(
                        [scaled_depth_maps_1, warped_depth_maps_2_to_1,
                         intersect_masks_1, intrinsics]) + depth_consistency_loss_function(
                        [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, intrinsics]))

                    loss = depth_consistency_loss + sparse_flow_loss
                    tq.update(batch_size)
                    if not np.isnan(loss.item()):
                        if batch == 0:
                            mean_loss = loss.item()
                            mean_depth_consistency_loss = depth_consistency_loss.item()
                            mean_sparse_flow_loss = sparse_flow_loss.item()
                        else:
                            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
                            mean_depth_consistency_loss = (mean_depth_consistency_loss * batch +
                                                           depth_consistency_loss.item()) / (batch + 1.0)
                            mean_sparse_flow_loss = (mean_sparse_flow_loss * batch + sparse_flow_loss.item()) / (
                                    batch + 1.0)

                    # Display depth and color at TensorboardX
                    if batch % display_each == 0:
                        colors_1_display, pred_depths_1_display, sparse_flows_1_display, dense_flows_1_display = \
                            utils.display_color_depth_sparse_flow_dense_flow(1, step, writer, colors_1,
                                                                             scaled_depth_maps_1 * boundaries,
                                                                             sparse_flows_1, flows_from_depth_1,
                                                                             phase="Validation", is_return_image=True,
                                                                             color_reverse=True)
                        colors_2_display, pred_depths_2_display, sparse_flows_2_display, dense_flows_2_display = \
                            utils.display_color_depth_sparse_flow_dense_flow(2, step, writer, colors_2,
                                                                             scaled_depth_maps_2 * boundaries,
                                                                             sparse_flows_2, flows_from_depth_2,
                                                                             phase="Validation", is_return_image=True,
                                                                             color_reverse=True)
                        utils.stack_and_display(phase="Validation",
                                                title="Results (c1, d1, sf1, df1, c2, d2, sf2, df2)",
                                                step=step, writer=writer,
                                                image_list=[colors_1_display, pred_depths_1_display,
                                                            sparse_flows_1_display,
                                                            dense_flows_1_display,
                                                            colors_2_display, pred_depths_2_display,
                                                            sparse_flows_2_display,
                                                            dense_flows_2_display])

                    # TensorboardX
                    writer.add_scalars('Validation', {'overall': mean_loss,
                                                      'depth_consistency': mean_depth_consistency_loss,
                                                      'sparse_flow': mean_sparse_flow_loss}, epoch)

            tq.close()
            model_path_epoch = log_root / 'checkpoint_model_epoch_{}_validation_{}.pt'.format(epoch,
                                                                                              mean_sparse_flow_loss)
            utils.save_model(model=depth_estimation_model, optimizer=optimizer,
                             epoch=epoch + 1, step=step, model_path=model_path_epoch,
                             validation_loss=mean_sparse_flow_loss)

        writer.close()
