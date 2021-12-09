import torch
from torch import nn


class SparseMaskedL1Loss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(SparseMaskedL1Loss, self).__init__()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        flows, flows_from_depth, sparse_masks = x
        loss = torch.sum(sparse_masks * torch.abs(flows - flows_from_depth),
                         dim=(1, 2, 3)) / (self.epsilon + torch.sum(sparse_masks, dim=(1, 2, 3)))
        return torch.mean(loss)


class NormalizedDistanceLoss(nn.Module):
    def __init__(self, height, width, eps=1.0e-5):
        super(NormalizedDistanceLoss, self).__init__()
        self.eps = eps
        self.y_grid, self.x_grid = torch.meshgrid(
            [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
             torch.arange(start=0, end=width, dtype=torch.float32).cuda()])
        self.y_grid = self.y_grid.reshape(1, 1, height, width)
        self.x_grid = self.x_grid.reshape(1, 1, height, width)

    def forward(self, x):
        depth_maps, warped_depth_maps, intersect_masks, intrinsics = x
        fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
        fy = intrinsics[:, 1, 1].reshape(-1, 1, 1, 1)
        cx = intrinsics[:, 0, 2].reshape(-1, 1, 1, 1)
        cy = intrinsics[:, 1, 2].reshape(-1, 1, 1, 1)

        with torch.no_grad():
            mean_value = torch.sum(intersect_masks * depth_maps, dim=(1, 2, 3), keepdim=False) / (
                    self.eps + torch.sum(intersect_masks, dim=(1, 2, 3),
                                         keepdim=False))

        location_3d_maps = torch.cat(
            [(self.x_grid - cx) / fx * depth_maps, (self.y_grid - cy) / fy * depth_maps, depth_maps], dim=1)

        warped_location_3d_maps = torch.cat(
            [(self.x_grid - cx) / fx * warped_depth_maps, (self.y_grid - cy) / fy * warped_depth_maps,
             warped_depth_maps], dim=1)

        loss = 2.0 * torch.sum(intersect_masks * torch.abs(location_3d_maps - warped_location_3d_maps), dim=(1, 2, 3),
                               keepdim=False) / \
               (1.0e-5 * mean_value + torch.sum(
                   intersect_masks * (depth_maps + torch.abs(warped_depth_maps)), dim=(1, 2, 3),
                   keepdim=False))
        return torch.mean(loss)
