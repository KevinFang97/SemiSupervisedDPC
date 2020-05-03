import torch
import torch.nn as nn
import numpy as np
from ssim import ssim

def compute_gradient_mask(img, thresh=0.03):
    img = img.mean(dim=1)
    D_dx = img[:, :, 1:] - img[:, :, :-1]
    D_dy = img[:, 1:] - img[:, :-1]
    padding_x = torch.zeros(img.size(0), img.size(1),1).type_as(img)
    padding_y = torch.zeros(img.size(0), 1, img.size(2)).type_as(img)
    D_dx = torch.cat((D_dx, padding_x), dim=2)
    D_dy = torch.cat((D_dy, padding_y), dim=1)

    mask = 0.5*(D_dx.abs() + D_dy.abs()) > thresh
    mask = mask.unsqueeze(1).float()
    return mask

def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x

class photometric_reconstruction_loss(nn.Module):
    def __init__(self):
        super(photometric_reconstruction_loss, self).__init__()

    def forward(self, input_img, output_img, exp_mask=None, pose_vec_weight=None, validate=False):
        B,_,h,w = input_img.size()
        output_img = nn.functional.adaptive_avg_pool2d(output_img, (h, w))
        reconstruction_loss = 0
        out_of_bound = 1 - (input_img == 0).prod(1, keepdim=True).type_as(input_img)
        diff = (output_img - input_img).abs() * out_of_bound
        if exp_mask is not None and validate == False:
            diff=diff*(exp_mask.expand_as(diff)) #expand_as makes the 1 channel mask 3 channels if imgs are colour
        if pose_vec_weight is not None:
            pose_vec_mask = pose_vec_weight[:,3:6].norm(dim=1) >=0.005
            pose_vec_mask = pose_vec_mask.cpu().detach().numpy().reshape((-1,1))
            pose_vec_mask = np.repeat(pose_vec_mask, output_img.size(2),axis=1)
            pose_vec_mask = np.repeat(pose_vec_mask.reshape((-1, output_img.size(2),1)),output_img.size(3),axis=2)
            pose_vec_mask = torch.FloatTensor(pose_vec_mask).unsqueeze(1).expand_as(output_img)
            if validate == False:
                diff = diff + 4*diff*(pose_vec_mask.type_as(output_img))
            if validate == True:
                gradient_mask = compute_gradient_mask(output_img)
                diff = diff*gradient_mask.expand_as(input_img)
                diff = diff*(pose_vec_mask.type_as(output_img))
        reconstruction_loss += diff.mean()
        assert((reconstruction_loss == reconstruction_loss).item() == 1)
        return reconstruction_loss


class photometric_reconstruction_loss_with_ssim(nn.Module):
    def __init__(self):
        super(photometric_reconstruction_loss_with_ssim, self).__init__()

    def forward(self, input_img, output_img, exp_mask=None, pose_vec_weight=None, validate=False):
        B,_,h,w = input_img.size()
        output_img = nn.functional.adaptive_avg_pool2d(output_img, (h, w))
        reconstruction_loss = 0
        out_of_bound = 1 - (input_img == 0).prod(1, keepdim=True).type_as(input_img)
        ssim_loss = 1 - ssim(output_img, input_img) * out_of_bound
        diff = (output_img - input_img) * out_of_bound
        if exp_mask is not None and validate == False:
            diff=diff*(exp_mask.expand_as(diff)) #expand_as makes the 1 channel mask 3 channels if imgs are colour
            ssim_loss = ssim_loss*(exp_mask.expand_as(ssim_loss))
        if pose_vec_weight is not None:
            pose_vec_mask = pose_vec_weight[:,3:6].norm(dim=1) >=0.005
            pose_vec_mask = pose_vec_mask.cpu().detach().numpy().reshape((-1,1))
            pose_vec_mask = np.repeat(pose_vec_mask, output_img.size(2),axis=1)
            pose_vec_mask = np.repeat(pose_vec_mask.reshape((-1, output_img.size(2),1)),output_img.size(3),axis=2)
            pose_vec_mask = torch.FloatTensor(pose_vec_mask).unsqueeze(1)#.expand_as(output_img)
            if validate == False:
                diff = diff + 4*diff*(pose_vec_mask.expand_as(diff).type_as(output_img))
                ssim_loss = ssim_loss + 4*ssim_loss*(pose_vec_mask.expand_as(ssim_loss).type_as(ssim_loss))
            if validate == True:
                gradient_mask = compute_gradient_mask(output_img)
                diff = diff*gradient_mask.expand_as(input_img)
                diff = diff*(pose_vec_mask.expand_as(diff).type_as(output_img))
                ssim_loss = ssim_loss*gradient_mask.expand_as(ssim_loss)
                ssim_loss = ssim_loss*(pose_vec_mask.expand_as(ssim_loss).type_as(ssim_loss))
        reconstruction_loss += 0.5 * (robust_l1(diff) + 0.5 * ssim_loss.mean())
        assert((reconstruction_loss == reconstruction_loss).item() == 1)
        return reconstruction_loss


class explainability_loss(torch.nn.Module):
    def __init__(self):
        super(explainability_loss, self).__init__()

    def forward(self, mask):
        if type(mask) not in [tuple, list]:
            mask = [mask]
        loss = 0
        for mask_scaled in mask:
            ones_var = torch.ones(1).expand_as(mask_scaled).type_as(mask_scaled)
            loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
        return loss

class smoothness_loss(torch.nn.Module):
    # ref: https://github.com/anuragranj/cc
    def __init__(self):
        super(smoothness_loss, self).__init__()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        return gy

    def get_edge_smoothness(self, img, pred):
        pred_gradients_x = self.gradient_x(pred)
        pred_gradients_y = self.gradient_y(pred)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    def forward(self, img, pred_disp):
        loss = 0
        weight = 1.
        for scaled_disp in pred_disp:
            b, _, h, w = scaled_disp.size()
            scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
            loss += self.get_edge_smoothness(scaled_img, scaled_disp)
            weight /= 2.3   # 2sqrt(2)

        return loss


class Compute_Loss(nn.modules.Module):
    def __init__(self, spatial_trans, photometric_loss, exp_loss, exp_weight=0.08, smo_weight=0.005, supervise_weight=2):
        super(Compute_Loss, self).__init__()
        self.spatial_trans = spatial_trans
        self.photometric_loss = photometric_loss
        self.smo_loss = smoothness_loss()
        self.exp_loss = exp_loss
        self.exp_weight = exp_weight
        self.smo_weight = smo_weight
        self.supervise_weight = supervise_weight
        
    def forward(self, current_img, target_img, pose, exp_mask, disparities, intrinsics, pose_vec_weight=None, validate=False, corr=None, gt_corr=None):
        loss = 0
        depth = [1.0/disp for disp in disparities]
        current_pose = -pose.clone()
        for d, m in zip(depth,exp_mask):
            x_reconstructed = self.spatial_trans(current_img, d, current_pose, intrinsics, intrinsics.inverse())
            loss += self.photometric_loss(x_reconstructed, target_img, exp_mask=m, pose_vec_weight=pose_vec_weight, validate=validate)

        if exp_mask[0] is not None and validate==False:
            loss += self.exp_weight*self.exp_loss(exp_mask)
        
        #print(corr)
        #print(gt_corr)
        #print(loss)
        #loss, su_loss, smo_loss before training: 0.5829, 0.1733, 99.1567
        if corr is not None:
            supervised_loss = torch.mean(torch.norm(corr[:,:3]-gt_corr[:,:3],dim=1) + torch.norm(corr[:,3:] - gt_corr[:,3:],dim=1))
            #print(supervised_loss)
            loss += self.supervise_weight * supervised_loss

        smooth_loss = self.smo_weight*self.smo_loss(target_img, depth)
        #print(smooth_loss)
        loss += smooth_loss
        #quit()

        return loss

if __name__ == '__main__':
    # loss = photometric_reconstruction_loss_with_ssim()
    loss = smoothness_loss()
    input_img = torch.rand([3,3,20,20], requires_grad=True)
    input_disp = [torch.rand([3,3,2,2], requires_grad=True),
                  torch.rand([3,3,5,5], requires_grad=True),
                  torch.rand([3,3,10,10], requires_grad=True)]
    # output_img = torch.rand([3,3,10,10], requires_grad=True)
    # exp_mask = torch.rand([3,1,10,10], requires_grad=True)
    # pose_vec_weight = torch.rand([3,6], requires_grad=True)
    # validate = False
    # res = loss(input_img, output_img, exp_mask, pose_vec_weight, validate)
    res = loss(input_img, input_disp)
    print(0.05*res)
