import torch
from torch import nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop_(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


def center_crop_(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class DepthScalingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthScalingLayer, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()
        self.zero = torch.tensor(0.0).float().cuda()
        self.one = torch.tensor(1.0).float().cuda()

    def forward(self, x):
        absolute_depth_estimations, input_sparse_depths, input_weighted_sparse_masks = x
        # Use sparse depth values which are greater than a certain ratio of the mean value of the sparse depths to avoid
        # unstability of scale recovery
        input_sparse_binary_masks = torch.where(input_weighted_sparse_masks > 1.0e-8, self.one, self.zero)
        mean_sparse_depths = torch.sum(input_sparse_depths * input_sparse_binary_masks, dim=(1, 2, 3),
                                       keepdim=True) / torch.sum(input_sparse_binary_masks, dim=(1, 2, 3), keepdim=True)
        above_mean_masks = torch.where(input_sparse_depths > 0.5 * mean_sparse_depths, self.one, self.zero)

        # Introduce a criteria to reduce the variation of scale maps
        sparse_scale_maps = input_sparse_depths * above_mean_masks / (self.epsilon + absolute_depth_estimations)
        mean_scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3), keepdim=True) / torch.sum(above_mean_masks,
                                                                                            dim=(1, 2, 3), keepdim=True)
        centered_sparse_scale_maps = sparse_scale_maps - above_mean_masks * mean_scales
        scale_stds = torch.sqrt(torch.sum(centered_sparse_scale_maps * centered_sparse_scale_maps, dim=(1, 2, 3),
                                          keepdim=False) / torch.sum(above_mean_masks, dim=(1, 2, 3), keepdim=False))
        scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3)) / torch.sum(above_mean_masks, dim=(1, 2, 3))
        return torch.mul(scales.reshape(-1, 1, 1, 1), absolute_depth_estimations), torch.mean(scale_stds / mean_scales)


class FlowfromDepthLayer(torch.nn.Module):
    def __init__(self):
        super(FlowfromDepthLayer, self).__init__()

    def forward(self, x):
        depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        flow_image = _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                      intrinsic_matrices)
        return flow_image


def _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices):
    # Generate a meshgrid for each depth map to calculate value
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.reshape(1, height, width, 1)
    y_grid = y_grid.reshape(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda()

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda().reshape(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.solve(eye, intrinsic_matrices)

    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).reshape(height, width, 3, 1)
    intermediate_result = torch.matmul(M.reshape(-1, 1, 1, 3, 3), mesh_grid).reshape(-1, height, width, 3)

    depth_maps_2_calculate = W.reshape(-1, 3).narrow(dim=-1, start=2, length=1).reshape(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).reshape(-1, height,
                                                                      width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.tensor(1.0e30).float().cuda() * (torch.tensor(1.0).float().cuda() - img_masks) + \
                             img_masks * depth_maps_2_calculate

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.reshape(-1, 3).narrow(dim=-1, start=0, length=1).reshape(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                               intermediate_result.narrow(
                                                                                                   dim=-1, start=0,
                                                                                                   length=1).reshape(-1,
                                                                                                                     height,
                                                                                                                     width,
                                                                                                                     1))) / depth_maps_2_calculate

    v_2 = (W.reshape(-1, 3).narrow(dim=-1, start=1, length=1).reshape(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                               intermediate_result.narrow(
                                                                                                   dim=-1, start=1,
                                                                                                   length=1).reshape(-1,
                                                                                                                     height,
                                                                                                                     width,
                                                                                                                     1))) / depth_maps_2_calculate
    return [u_2, v_2]


# Optical flow for frame 1 to frame 2
def _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices):
    # BxHxWxC
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.reshape(1, height, width, 1)
    y_grid = y_grid.reshape(1, height, width, 1)

    u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                         intrinsic_matrices)

    return torch.cat(
        [(u_2 - x_grid) / torch.tensor(width).float().cuda(), (v_2 - y_grid) / torch.tensor(height).float().cuda()],
        dim=-1).permute(0, 3, 1, 2)


class DepthWarpingLayer(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthWarpingLayer, self).__init__()
        self.zero = torch.tensor(0.0).float().cuda()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        warped_depth_maps, intersect_masks = _depth_warping(depth_maps_1, depth_maps_2, img_masks,
                                                            translation_vectors,
                                                            rotation_matrices, intrinsic_matrices, self.epsilon)
        return warped_depth_maps, intersect_masks


# Warping depth map in coordinate system 2 to coordinate system 1
def _depth_warping(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                   intrinsic_matrices, epsilon):
    # Generate a meshgrid for each depth map to calculate value
    # BxHxWxC
    depth_maps_1 = torch.mul(depth_maps_1, img_masks)
    depth_maps_2 = torch.mul(depth_maps_2, img_masks)

    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)

    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.reshape(1, height, width, 1)
    y_grid = y_grid.reshape(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda()

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda().reshape(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.solve(eye, intrinsic_matrices)
    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).reshape(height, width, 3, 1)
    intermediate_result = torch.matmul(M.reshape(-1, 1, 1, 3, 3), mesh_grid).reshape(-1, height, width, 3)

    depth_maps_2_calculate = W.reshape(-1, 3).narrow(dim=-1, start=2, length=1).reshape(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).reshape(-1, height,
                                                                      width, 1))
    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.where(img_masks > 0.5, depth_maps_2_calculate, epsilon)
    depth_maps_2_calculate = torch.where(depth_maps_2_calculate > 0.0, depth_maps_2_calculate, epsilon)

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.reshape(-1, 3).narrow(dim=-1, start=0, length=1).reshape(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                               intermediate_result.narrow(
                                                                                                   dim=-1, start=0,
                                                                                                   length=1).reshape(-1,
                                                                                                                     height,
                                                                                                                     width,
                                                                                                                     1))) / (
              depth_maps_2_calculate)

    v_2 = (W.reshape(-1, 3).narrow(dim=-1, start=1, length=1).reshape(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                               intermediate_result.narrow(
                                                                                                   dim=-1, start=1,
                                                                                                   length=1).reshape(-1,
                                                                                                                     height,
                                                                                                                     width,
                                                                                                                     1))) / (
              depth_maps_2_calculate)

    W_2 = torch.bmm(intrinsic_matrices, translation_vectors)
    M_2 = torch.bmm(torch.bmm(intrinsic_matrices, rotation_matrices), intrinsic_matrices_inverse)

    temp = torch.matmul(M_2.reshape(-1, 1, 1, 3, 3), mesh_grid).reshape(-1, height, width, 3).narrow(dim=-1, start=2,
                                                                                                     length=1).reshape(
        -1,
        height,
        width, 1)
    depth_maps_1_calculate = W_2.reshape(-1, 3).narrow(dim=-1, start=2, length=1).reshape(-1, 1, 1, 1) + torch.mul(
        depth_maps_2, temp)
    depth_maps_1_calculate = torch.mul(img_masks, depth_maps_1_calculate)

    u_2_flat = u_2.reshape(-1)
    v_2_flat = v_2.reshape(-1)

    warped_depth_maps_2 = _bilinear_interpolate(depth_maps_1_calculate, u_2_flat, v_2_flat).reshape(num_batch, 1,
                                                                                                    height,
                                                                                                    width)
    # binarize
    intersect_masks = torch.where(_bilinear_interpolate(img_masks, u_2_flat, v_2_flat) * img_masks >= 0.9,
                                  torch.tensor(1.0).float().cuda(),
                                  torch.tensor(0.0).float().cuda()).reshape(num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]


def _bilinear_interpolate(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    # Range [-1, 1]
    grid = torch.cat([torch.tensor(2.0).float().cuda() *
                      (x.reshape(num_batch, height, width, 1) / torch.tensor(width).float().cuda())
                      - torch.tensor(1.0).float().cuda(), torch.tensor(2.0).float().cuda() * (
                              y.reshape(num_batch, height, width, 1) / torch.tensor(
                          height).float().cuda()) - torch.tensor(
        1.0).float().cuda()], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode).permute(0, 2, 3, 1)