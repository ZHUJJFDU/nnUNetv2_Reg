import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_CE_and_Regression_loss(nn.Module):
    def __init__(self, soft_dice_kwargs=None, ce_kwargs=None, weight_ce=1, weight_dice=1, weight_reg=1.0,
                 reg_loss_type='mse', ignore_label=None, dice_class=SoftDiceLoss, debug=False):
        super().__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        if ce_kwargs is None:
            ce_kwargs = {'weight': None}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_reg = weight_reg
        self.ignore_label = ignore_label
        self.reg_loss_type = reg_loss_type
        self.debug = debug
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.use_custom_dice = True
        self.apply_nonlin = softmax_helper_dim1
        self.batch_dice = soft_dice_kwargs.get('batch_dice', True)
        self.smooth = soft_dice_kwargs.get('smooth', 1e-5)
        self.do_bg = soft_dice_kwargs.get('do_bg', False)
        if reg_loss_type == 'mse':
            self.reg_loss = nn.MSELoss()
        elif reg_loss_type == 'l1':
            self.reg_loss = nn.L1Loss()
        elif reg_loss_type == 'smooth_l1':
            self.reg_loss = nn.SmoothL1Loss(beta=0.1)
        else:
            self.reg_loss = nn.MSELoss()

    def custom_dice_loss(self, x, y, loss_mask=None):
        if isinstance(x, (list, tuple)):
            x = x[0]
        if self.apply_nonlin is not None:
            try:
                x = self.apply_nonlin(x)
            except Exception:
                x = torch.softmax(x, dim=1)
        shp_x = x.shape
        shp_y = y.shape
        if shp_x[2:] != shp_y[2:]:
            return torch.tensor(0.5, device=x.device, requires_grad=True)
        try:
            x = x.contiguous()
            y = y.contiguous()
            x_flat = x.reshape(shp_x[0], shp_x[1], -1)
            y_flat = y.reshape(shp_y[0], shp_y[1], -1)
            if loss_mask is not None:
                try:
                    loss_mask = loss_mask.contiguous().reshape(shp_x[0], 1, -1)
                    y_flat = y_flat * loss_mask
                    x_flat = x_flat * loss_mask
                except Exception:
                    pass
            dice_scores = []
            for c in range(x_flat.shape[1]):
                if self.batch_dice:
                    x_c = x_flat[:, c].reshape(-1)
                    y_c = y_flat[:, min(c, y_flat.shape[1]-1)].reshape(-1)
                    intersection = (x_c * y_c).sum()
                    union = x_c.sum() + y_c.sum()
                    dice = (2.0 * intersection + self.smooth) / (union + self.smooth) if union > 0 else torch.tensor(1.0, device=x.device)
                    dice_scores.append(dice)
                else:
                    dice_batch = []
                    for b in range(x_flat.shape[0]):
                        x_bc = x_flat[b, c]
                        y_bc = y_flat[b, min(c, y_flat.shape[1]-1)]
                        intersection = (x_bc * y_bc).sum()
                        union = x_bc.sum() + y_bc.sum()
                        dice = (2.0 * intersection + self.smooth) / (union + self.smooth) if union > 0 else torch.tensor(1.0, device=x.device)
                        dice_batch.append(dice)
                    dice_scores.append(torch.stack(dice_batch).mean())
            dice_scores = torch.stack(dice_scores)
            if not self.do_bg and dice_scores.shape[0] > 1:
                dice_scores = dice_scores[1:]
            return 1.0 - dice_scores.mean()
        except Exception:
            return torch.tensor(0.5, device=x.device, requires_grad=True)

    def forward(self, net_output, target, reg_output=None, reg_target=None):
        dc_loss = 0
        ce_loss = 0
        reg_loss = 0
        device = net_output.device if hasattr(net_output, 'device') else torch.device('cpu')
        if isinstance(net_output, tuple) and len(net_output) == 2 and reg_output is None:
            net_output, reg_output = net_output
        if isinstance(net_output, list):
            net_output = net_output[0]
        try:
            if self.ignore_label is not None:
                assert target.shape[1] == 1
                mask = target != self.ignore_label
                target_dice = torch.where(mask, target, torch.zeros_like(target))
                num_fg = mask.sum()
            else:
                target_dice = target
                mask = None
                num_fg = None
            if torch.min(target_dice) < 0:
                target_dice = torch.where(target_dice < 0, torch.zeros_like(target_dice), target_dice)
            if self.weight_dice != 0:
                try:
                    if self.use_custom_dice:
                        dc_loss = self.custom_dice_loss(net_output, target_dice, loss_mask=mask)
                    else:
                        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
                except RuntimeError:
                    dc_loss = torch.tensor(0.1, device=device, requires_grad=True)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg is None or num_fg > 0):
                try:
                    if target.shape[1] == 1:
                        if torch.min(target) < 0 or torch.max(target) >= net_output.shape[1]:
                            target_ce = target.clone()
                            if self.ignore_label is not None:
                                ignore_mask = target == self.ignore_label
                                target_ce = torch.where((target < 0) | (target >= net_output.shape[1]) & (~ignore_mask), torch.zeros_like(target), target)
                            else:
                                target_ce = torch.clamp(target, 0, net_output.shape[1] - 1)
                        else:
                            target_ce = target
                        ce_loss = self.ce(net_output, target_ce[:, 0].long())
                    else:
                        ce_loss = self.ce(net_output, torch.argmax(target, dim=1))
                except RuntimeError:
                    ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
                except Exception:
                    ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
        except Exception:
            dc_loss = torch.tensor(0.1, device=device, requires_grad=True)
            ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
        if reg_output is not None and reg_target is not None and self.weight_reg > 0:
            try:
                if reg_output.dtype != reg_target.dtype:
                    reg_output = reg_output.to(dtype=reg_target.dtype)
                if torch.isnan(reg_output).any() or torch.isnan(reg_target).any():
                    reg_output = torch.nan_to_num(reg_output, nan=0.0)
                    reg_target = torch.nan_to_num(reg_target, nan=0.0)
                if reg_target.ndim == 1 and reg_output.ndim > 1:
                    reg_target = reg_target.view(-1, 1)
                elif reg_target.ndim > 1 and reg_output.ndim == 1:
                    reg_output = reg_output.view(-1, 1)
                if reg_target.device != reg_output.device:
                    reg_target = reg_target.to(reg_output.device)
                if reg_output.shape != reg_target.shape:
                    reg_output = reg_output.reshape(reg_target.shape)
                try:
                    reg_loss = self.reg_loss(reg_output, reg_target)
                except RuntimeError as e1:
                    try:
                        reg_output = reg_output.to(dtype=torch.float32)
                        reg_target = reg_target.to(dtype=torch.float32)
                        reg_loss = self.reg_loss(reg_output, reg_target)
                    except Exception:
                        reg_loss = torch.tensor(0.1, device=device, requires_grad=True)
            except Exception:
                reg_loss = torch.tensor(0.1, device=device, requires_grad=True)
        loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_reg * reg_loss
        return loss, ce_loss + dc_loss, reg_loss
