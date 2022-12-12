import torch
import torch.nn as nn

class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val = 'middle', spacing='random', transforms_like=False, p_random_apply_rand_quant = 1):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)

        # region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x

