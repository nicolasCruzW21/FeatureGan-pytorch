import torch
import torchvision
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
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

#TODO: cambiar la textura del piso del field
class TensorGeneratorModel(BaseModel):

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
            visual_names_B = ['input', 'input_blured','idt'] #,'real_B', 'real_A_mask','real_B_mask', 'real_A_depth','real_B_depth', 'real_A_blured','real_B_blured'
            self.loss_names = ['G', 'idt_A']

            #visual_names_B = ['real_A','real_B','seg_cars_target','seg_cars', 'fake_A', 'fake_A_D','lines_real_B', 'lines_fake_A', 'robots_real_B', 'robots_fake_A', 'field_real_B', 'field_fake_A', 'shirt_real_B', 'shirt_fake_A','squeeze_real_M1','squeeze_fake_M1']
            #self.loss_names = ['D_B_G', 'D_B_M', 'G_B','G_B_M', 'G_B_G', 'F_B', 'F_B_robots', 'F_B_field', 'F_B_shirt', 'F_B_lines', 'G', 'idt_B']
        else:
            visual_names_B = ['real_B', 'fake_A']
            self.loss_names = ['idt_B']
        

        self.visual_names = visual_names_B

        if self.isTrain:
            self.model_names = ['G_B']#,'D_B_G']
        else:  
            self.model_names = ['G_B']

        self.netG_B = networks.define_G(1078+3, 3, 32, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators


            #self.netD_B_G= networks.define_D(320, 32, "n_layers", 
                                            #3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 0, False, True)

            self.netF = networks.define_F(self.gpu_ids)
            #self.netF=self.netF.eval()
            self.set_requires_grad([self.netF], False)


        else:
            self.netG_B=self.netG_B.eval()
            self.set_requires_grad([self.netG_B], False)
            self.loss_idt_B = 0

		
 
        self.lay0 = torch.nn.InstanceNorm2d(3, affine=False).cuda()
        self.lay1 = torch.nn.LayerNorm([3, opt.crop_size, opt.crop_size], elementwise_affine=False).cuda()
        self.sig = nn.Sigmoid()
        self.criterionSeg = torch.nn.BCEWithLogitsLoss()
        self.upsample_Feature = torch.nn.Upsample(size=256, mode='bilinear').cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            factorAll = 0.1

            self.criterionFeatureRobots = networks.FeatureLoss(1*factorAll ,      4*factorAll,      2*factorAll,      "L1").to(self.device)

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            self.normalizer128 = torch.nn.InstanceNorm2d(64)
            self.normalizer256 = torch.nn.InstanceNorm2d(256)
            
            self.reflect = nn.ReflectionPad2d(2)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=19, stride=1, padding=9)

            self.avg_pool_disc = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

            self.upsample_L = torch.nn.Upsample(size=opt.crop_size, mode='nearest')
            self.upsample_L_seg = torch.nn.Upsample(size=opt.crop_size, mode='nearest')

            self.upsample_M = torch.nn.Upsample(size=(int)(512), mode='nearest')
            self.upsample_G = torch.nn.Upsample(size=(int)(96), mode='nearest')
            self.upsample_crop = torch.nn.Upsample(size=(int)(opt.crop_size), mode='nearest')
            disc_size = 30
            self.downsample_disc = torch.nn.Upsample(size=(int)(disc_size), mode='nearest')
            self.ones_disc = torch.ones(opt.crop_size, opt.crop_size).cuda()
            self.jitter = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.02)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr*1, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            
            self.aa=np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
            self.bb=torch.autograd.variable(torch.from_numpy(self.aa).float().permute(0,3,1,2).cuda())
            self.Processed = False



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

        self.model = input['A_models' if AtoB else 'B_models'].to(self.device)
        self.model = self.model.squeeze(0)
        #print("shape",self.model.shape)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.FeaturesCalculated = True

    def calculate_Features(self, image):
        
        generated = (image+1.0)/2.0*255.0
        input_image = generated-self.bb
        return self.netF(input_image)

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        
        if(self.Processed == False):
            out0, out1, out2 = self.calculate_Features(self.upsample_Feature(self.model[0,:,:,:].unsqueeze(0)))
            concat = torch.cat([ self.upsample_M(out0[0,:,:,:].unsqueeze(0)), self.upsample_M(out1[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)        
            tensor = concat
            #print("shape2",self.model.shape)

            for i in range(1,self.model.shape[0]):
                
                out0, out1, out2 = self.calculate_Features(self.upsample_Feature(self.model[i,:,:,:].unsqueeze(0)))
                concat = torch.cat([ self.upsample_M(out0[0,0:10,:,:].unsqueeze(0)), self.upsample_M(out1[0,0:10,:,:].unsqueeze(0)), self.upsample_M(out2[0,0:10,:,:].unsqueeze(0))], 1)
                tensor = torch.cat([tensor, concat], 1)
                print(tensor.shape)
            self.tensor = tensor
            self.Processed = True
        
        lambda_idt = self.opt.lambda_identity
        self.input = self.real_A*self.real_A_mask
        self.input_blured = self.avg_pool(self.real_A)*self.real_A_mask
        inToNet = torch.cat([self.input_blured, self.tensor], 1)#self.reflect
        #print(inToNet.shape)
        self.idt, squeezed = self.netG_B(inToNet)
        self.loss_idt_A = self.criterionL1(self.idt, self.input)*10
        print(self.loss_idt_A)




        

#--------------------------------Total------------------------------------------------------

        self.loss_G = self.loss_idt_A



    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        #self.set_requires_grad([self.netG_B], True)

        self.forward()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights

        #self.optimizer_M.step()

        

