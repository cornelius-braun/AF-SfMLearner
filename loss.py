import torch
import torch.nn as nn
import torch.nn.functional as F
#from Models.rand_models import DiscExtractorModule, FeatureExtractorModule

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class MonodepthLoss(nn.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)
        
        self.left_pyramid= left_pyramid
        self.right_pyramid= right_pyramid
        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss

class GANLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.device= torch.device('cuda:{}'.format(self.args.device))
        if self.args.gan_type=='ls':
            self.loss_fnc= nn.MSELoss().to(self.device)
        elif self.args.gan_type=='bce':
            self.loss_fnc= nn.BCELoss().to(self.device)
        else:
            raise Exception('gan loss type: {} not defined'.format(self.gan_type))
    
    def forward(self, pred, gt):
        return self.loss_fnc(pred, gt)

class PerceptualLoss(nn.Module):
    def __init__(self, args, n=4, STYLE_w=100.0):
        super(PerceptualLoss, self).__init__()
        self.PERC_w= args.perc_w
        self.STYLE_w= STYLE_w
        self.style_bool=args.style_bool
        #self.extractor= extractor # VGG extractor
        self.args= args
        self.n= n
        self.device= torch.device('cuda:{}'.format(self.args.device))
        self.l1= nn.L1Loss().to(self.device)
        if self.args.perc_type=='feat' or self.args.perc_type=='both':
            self.pretrained_extractor= FeatureExtractorModule(self.args.perc_type)
        else:
            self.pretrained_extractor= None

    def get_feature_maps(self, left_gt, right_gt, left_est, right_est, disc_extractor):

        if disc_extractor!= None and self.args.perc_type=='disc':
            gt_ft= disc_extractor(left_gt, right_gt)
            est_ft= disc_extractor(left_est, right_est)
            return gt_ft, est_ft

        elif self.args.perc_type=='feat':
            left_gt_ft= [self.pretrained_extractor(left_gt[i]) for i in range(self.n)]
            right_gt_ft= [self.pretrained_extractor(right_gt[i]) for i in range(self.n)]
            left_est_ft= [self.pretrained_extractor(left_est[i]) for i in range(self.n)]
            right_est_ft= [self.pretrained_extractor(right_est[i]) for i in range(self.n)]
        
            return left_gt_ft, right_gt_ft, left_est_ft, right_est_ft
        
        elif self.args.perc_type=='both':
            gt_disc_ft= disc_extractor(left_gt, right_gt)
            est_disc_ft= disc_extractor(left_est, right_est)
            left_gt_feat_ft= [self.pretrained_extractor(left_gt[i]) for i in range(self.n)]
            right_gt_feat_ft= [self.pretrained_extractor(right_gt[i]) for i in range(self.n)]
            left_est_feat_ft= [self.pretrained_extractor(left_est[i]) for i in range(self.n)]
            right_est_feat_ft= [self.pretrained_extractor(right_est[i]) for i in range(self.n)]
            return gt_disc_ft, est_disc_ft, left_gt_feat_ft, right_gt_feat_ft, left_est_feat_ft, right_est_feat_ft

    """def forward(self, left_est, right_est, left_pyramid, right_pyramid, disc_model):
        
        if self.args.perc_type=='disc':
            disc_extractor= DiscExtractorModule(disc_model)
            # Perceptual Loss
            gt_ft, est_ft= self.get_feature_maps(left_pyramid, right_pyramid, left_est, right_est, disc_extractor)
            perc = 0.0
            for i in range(self.n):
                perc+= self.l1(est_ft[i], gt_ft[i])
            # add perceptual loss to image_loss!
            perc_loss= self.PERC_w * perc

            if self.style_bool:
                style=0.0
                for i in range(self.n):
                    style+= self.l1(gram_matrix(est_ft[i]), gram_matrix(gt_ft[i]))

                style_loss= self.STYLE_w * style
                
                return perc_loss, style_loss
            
            else:
                return perc_loss

        elif self.args.perc_type=='feat':
            disc_extractor=None
            # Perceptual Loss
            left_gt_ft, right_gt_ft, left_est_ft, right_est_ft= self.get_feature_maps(left_pyramid, right_pyramid, left_est, right_est, disc_extractor)
            perc_left = 0.0
            perc_right = 0.0
            for i in range(self.n):
                for j in range(3):
                    perc_left+= self.l1(left_est_ft[i][j], left_gt_ft[i][j])
                    perc_right+= self.l1(right_est_ft[i][j], right_gt_ft[i][j])

            # add perceptual loss to image_loss!
            perc_loss= self.PERC_w * (perc_left + perc_right)

            if self.style_bool:
                style_left= 0.0
                style_right=0.0
                for i in range(self.n):
                    for j in range(3):
                        style_left+= self.l1(gram_matrix(left_est_ft[i][j]), gram_matrix(left_gt_ft[i][j]))
                        style_right+= self.l1(gram_matrix(right_est_ft[i][j]), gram_matrix(right_gt_ft[i][j]))

                style_loss= self.STYLE_w * (style_left + style_right)
        
                return perc_loss, style_loss
            
            else:
                return perc_loss

        elif self.args.perc_type=='both':

            disc_extractor= DiscExtractorModule(disc_model)
            # Perceptual Discriminator Loss
            gt_disc_ft, est_disc_ft, left_gt_feat_ft, right_gt_feat_ft, left_est_feat_ft, right_est_feat_ft= self.get_feature_maps(left_pyramid, right_pyramid, left_est, right_est, disc_extractor)
            
            perc_disc = 0.0
            for i in range(self.n):
                perc_disc+= self.l1(est_disc_ft[i], gt_disc_ft[i])
            # add perceptual loss to image_loss!
            perc_disc_loss= self.PERC_w * perc_disc


            # Perceptual Feature Extractir Loss
            perc_left = 0.0
            perc_right = 0.0
            for i in range(self.n):
                for j in range(3):
                    perc_left+= self.l1(left_est_feat_ft[i][j], left_gt_feat_ft[i][j])
                    perc_right+= self.l1(right_est_feat_ft[i][j], right_gt_feat_ft[i][j])

            # add perceptual loss to image_loss!
            perc_feat_loss= self.PERC_w * (perc_left + perc_right)

            perc_loss= 0.5 * (perc_disc_loss + perc_feat_loss) 
            
            if self.style_bool:
                style_left= 0.0
                style_right=0.0
                for i in range(self.n):
                    for j in range(3):
                        style_left+= self.l1(gram_matrix(left_est_ft[i][j]), gram_matrix(left_gt_ft[i][j]))
                        style_right+= self.l1(gram_matrix(right_est_ft[i][j]), gram_matrix(right_gt_ft[i][j]))

                style_loss= self.STYLE_w * (style_left + style_right)
        
                return perc_loss, style_loss
            
            else:
                return perc_loss

        else:
            raise Exception('perceptual loss type: {} not defined'.format(self.args.perc_type))

class WasserrsteinGanLoss(nn.Module):

    def __init__(self, drift=0.001, use_gp=False):
        super().__init__()
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, reg_lambda=10):
     

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = self.dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def disc_loss(self, disc_real_out, disc_fake_out, real_imgs, fake_imgs):
        # define the (Wasserstein) loss
        #fake_out = self.dis(fake_samps)
        #real_out = self.dis(real_samps)

        loss = (torch.mean(disc_fake_out) - torch.mean(disc_real_out)
                + (self.drift * torch.mean(disc_real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_imgs, fake_imgs) #real_samps, fake_samps
            loss += gp

        return loss

    def gen_loss(self, disc_fake_out, real_label):
        # calculate the WGAN loss for generator
        loss = -torch.mean(disc_fake_out) # self.dis(fake_samps)

        return loss"""