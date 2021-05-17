import torch
import torchvision
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from .models import DPTGenModel, DPTDiscModel
from . import networks
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from util import util
from PIL import Image
import numpy as np
import time
from random import randrange
from statistics import mean
import torch.nn as nn
import random
import math  
from color_transfer import color_transfer
import cv2
from .transforms import Resize, NormalizeImage, PrepareForNet
#TODO: cambiar la textura del piso del field
class FeatureGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True) 
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        

        if self.isTrain:
            visual_names_B = ['real_A_D_S', 'pred_real', 'fake_A_D_S', 'pred_fake', 'real_B', 'fake_A', 'b']
            self.loss_names = ['G', 'G_B', 'F_B_B', 'F_B_F', 'F_B', 'D_B_M', 'D_fake', 'D_real']
        else:
            visual_names_B = ['real_B', 'fake_A']
            self.loss_names = ['idt_B']
        

        self.visual_names = visual_names_B

        if self.isTrain:
            self.model_names = ['G_B', 'D_B_M']#,'D_B_G']
        else:  
            self.model_names = ['G_B']

        self.netF = networks.define_F(self.gpu_ids)
        self.netF=self.netF.eval()
        #self.netF = self.netF.half()
        self.set_requires_grad([self.netF], False)


        net_w = net_h = 384
        self.netG_B = DPTGenModel(
            path=None,
            backbone="vitb16_384",
            non_negative=False,
            enable_attention_hooks=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.netG_B.to(device)


        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )


        #self.netG_B = networks.define_G(6, 3, 64, opt.netG, opt.norm,
                                        #not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators

            self.netD_B_M = networks.define_D(3, 64, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 1, False, False)

            #self.netD_B_M = DPTDiscModel(
            #path=None,
            #backbone="vitb16_384",
            #non_negative=False,
            #enable_attention_hooks=False,
            #)
        
            #self.netD_B_M.to(device)




        else:
            self.netG_B=self.netG_B.eval()
            self.set_requires_grad([self.netG_B], False)
            self.loss_idt_B = 0
        self.lay1 = torch.nn.LayerNorm([3, opt.crop_size, opt.crop_size], elementwise_affine=False).cuda()
        self.upsample_Feature = torch.nn.Upsample(size=512, mode='bilinear').cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.jitter = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05)
            self.factorForground = 0.2
            self.factorBackground = 5

            self.criterionFeatureRobots = networks.FeatureLoss(2 ,      4,      4,      "MSE").to(self.device)

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            self.upsample_L = torch.nn.Upsample(size=opt.crop_size, mode='nearest')
            self.upsample_L_seg = torch.nn.Upsample(size=opt.crop_size, mode='nearest')

            self.upsample_M = torch.nn.Upsample(size=(int)(512), mode='nearest')
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B_M.parameters()), lr=opt.lr*1)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr*1)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.aa=np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
            self.bb=torch.autograd.variable(torch.from_numpy(self.aa).float().permute(0,3,1,2).cuda())



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.real_A_depth = input['A_depth' if AtoB else 'B_depth'].to(self.device)
        self.real_B_depth = input['B_depth' if AtoB else 'A_depth'].to(self.device)

        self.real_A_mask = input['A_mask' if AtoB else 'B_mask'].to(self.device)
        self.real_B_mask = input['B_mask' if AtoB else 'A_mask'].to(self.device)

        #self.real_A_blured = input['A' if AtoB else 'B'].to(self.device)
        #self.real_B_blured = input['B' if AtoB else 'A'].to(self.device)

        #self.model = input['A_models' if AtoB else 'B_models'].to(self.device)
        #self.model = self.model.squeeze(0)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def processDPTSample(self, sample):
        numpy_img = util.tensor2im(sample)
        img_input = self.transform({"image": numpy_img})["image"]
        return torch.from_numpy(img_input).to(self.device).unsqueeze(0), numpy_img.shape[:2]

    def processDPTOutput(self, prediction, shape):
        return (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = shape,
                mode="bicubic",
                align_corners = False,
            )
            .squeeze()
        ).unsqueeze(0)


    def JitterSample(self, sample):
        PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(sample)))

        return util.im2tensor(np.array(PIL_real_A_Jitter))


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:


            sample, shape = self.processDPTSample(self.real_B)
            prediction = self.netG_B.forward(sample).squeeze(0)
            self.fake_A = self.processDPTOutput(prediction, shape)


        else:
            self.fake_A = self.netG_B(self.real_B)


    def backward_D_basic(self, real, fake, real_mask, fake_mask):
        #print(real.shape)
        #sample, shape = self.processDPTSample(real)
        #prediction = self.netD_B_M.forward(sample).squeeze(0)
        #pred_real = self.processDPTOutput(prediction.unsqueeze(0), shape)
        self.pred_real = self.upsample_M(self.netD_B_M(real))
        #print(self.pred_real.shape)
        #self.pred_real = pred_real.unsqueeze(0)
        
        self.loss_D_real = self.criterionGAN(self.pred_real, True, real_mask)



        #sample, shape = self.processDPTSample(fake.detach())
        #prediction = self.netD_B_M.forward(sample).squeeze(0)
        #pred_fake = self.processDPTOutput(prediction.unsqueeze(0), shape)
        #self.pred_fake = pred_fake.unsqueeze(0)
        self.pred_fake = self.upsample_M(self.netD_B_M(fake))
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False, fake_mask)
        
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        #print(loss_D)
        return loss_D#, self.lay1(squeeze_real[:,0:3, :]), self.lay1(squeeze_fake[:,0:3, :])

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        
        image_mask = self.fake_A_pool.query(torch.cat([self.fake_A, self.real_B_mask], 1))
        self.fake_A_D = self.JitterSample(image_mask[:,0:3,:,:])
        real_mask = image_mask[:,3,:,:]

        self.real_A_D = self.JitterSample(self.real_A)

        self.real_A_D_S = self.real_A_D * self.real_A_mask
        self.fake_A_D_S = self.fake_A_D * real_mask

        #toVGG = torch.cat([self.fake_A_D_S, self.real_A_D_S], 0)
        #out0, out1, out2 = self.calculate_Features(self.upsample_Feature(toVGG))

        #out0 = self.upsample_M(out0.float())
        #out1 = self.upsample_M(out1.float())
        #out2 = self.upsample_M(out2.float())

        #concat_fake_A_M = torch.cat([ out0[0,:,:,:].unsqueeze(0), out1[0,:,:,:].unsqueeze(0), out2[0,:,:,:].unsqueeze(0)], 1)

        #concat_real_A_M = torch.cat([ out0[1,:,:,:].unsqueeze(0), out1[1,:,:,:].unsqueeze(0), out2[1,:,:,:].unsqueeze(0)], 1)

        self.loss_D_B_M = self.backward_D_basic(self.real_A_D_S, self.fake_A_D_S, self.real_A_mask, real_mask)


    def calculate_Features(self, image):
        image = (image+1.0)/2.0*255.0-self.bb
        return self.netF((image))

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.opt.lambda_identity
        #if(self.opt.lambda_identity>0):
            #self.idt = self.netG_B(self.real_D)
            #self.loss_idt_A = self.criterionL1(self.idt, self.real_D)
            #self.loss_idt_A = lambda_idt * self.loss_idt_A
        #else:
        self.loss_idt_A = 0
        

        toVGG = torch.cat([self.fake_A*self.real_B_mask, self.real_B*self.real_B_mask], 0)
	
        out0, out1, out2 = self.calculate_Features(self.upsample_Feature(toVGG))
        out0 = self.upsample_M(out0.float())
        out1 = self.upsample_M(out1.float())
        out2 = self.upsample_M(out2.float())

        self.loss_F_B_F = self.criterionFeatureRobots(out0[1,:,:,:], out1[1,:,:,:], out2[1,:,:,:], out0[0,:,:,:], out1[0,:,:,:], out2[0,:,:,:]) * self.factorForground

        backgroundMask = ~(self.real_B_mask>0)
        self.fake_A_background = self.fake_A * backgroundMask
        self.real_B_background = self.real_B * backgroundMask


        self.loss_F_B_B = self.criterionL1(self.fake_A_background, self.real_B_background) * self.factorBackground
        self.loss_F_B = self.loss_F_B_B + self.loss_F_B_F



        #concat_fake_A_M = torch.cat([out0[0,:,:,:].unsqueeze(0), out1[0,:,:,:].unsqueeze(0), out2[0,:,:,:].unsqueeze(0)], 1)



        #self.b, squeeze =self.netD_B_M(concat_fake_A_M) 

        #sample, shape = self.processDPTSample(self.fake_A * self.real_B_mask)
        #prediction = self.netD_B_M.forward(sample).squeeze(0)
        #self.b = self.processDPTOutput(prediction.unsqueeze(0), shape)
        self.b = self.upsample_M(self.netD_B_M.forward(self.fake_A * self.real_B_mask))
        #print(self.b.shape)
        #self.b = self.b.unsqueeze(0)
        self.loss_G_B = self.criterionGAN(self.b, True, self.real_B_mask)

        self.loss_idt_B = 0



        

#--------------------------------Total------------------------------------------------------

        self.loss_G = self.loss_F_B + self.loss_idt_B + self.loss_G_B + self.loss_idt_A



    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.set_requires_grad([self.netD_B_M], True)#, self.netD_B_G], True)
        self.set_requires_grad([self.netG_B], False)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_B()      # calculate graidents for D_B
        self.loss_D_B_M.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights


        self.set_requires_grad([self.netD_B_M], False)#, self.netD_B_G], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_B], True)


        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights


