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
            visual_names_B = ['real_A', 'real_A_D_S', 'fake_A_D_S', 'squeeze_real', 'squeeze_fake','real_B', 'fake_A','real_B_depth', 'out_depth', 'model0', 'model1'] #,'real_B', 'real_A_mask','real_B_mask', 'real_A_depth','real_B_depth', 'real_A_blured','real_B_blured'
            self.loss_names = ['G', 'G_B', 'F_B', 'F_B_B', 'F_B_F', 'D_B_M']

            #visual_names_B = ['real_A','real_B','seg_cars_target','seg_cars', 'fake_A', 'fake_A_D','lines_real_B', 'lines_fake_A', 'robots_real_B', 'robots_fake_A', 'field_real_B', 'field_fake_A', 'shirt_real_B', 'shirt_fake_A','squeeze_real_M1','squeeze_fake_M1']
            #self.loss_names = ['D_B_G', 'D_B_M', 'G_B','G_B_M', 'G_B_G', 'F_B', 'F_B_robots', 'F_B_field', 'F_B_shirt', 'F_B_lines', 'G', 'idt_B']
        else:
            visual_names_B = ['real_B', 'fake_A']
            self.loss_names = ['idt_B']
        

        self.visual_names = visual_names_B

        if self.isTrain:
            self.model_names = ['G_B', 'D_B_M']#,'D_B_G']
        else:  
            self.model_names = ['G_B']

        self.netG_B = networks.define_G(4, 3, 64, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define discriminators

            self.netD_B_M = networks.define_D(448, 64, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 1, False, False)

            #self.netD_B_G= networks.define_D(320, 32, "n_layers", 
                                            #3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 0, False, True)

            self.netF = networks.define_F(self.gpu_ids)
            #self.netF=self.netF.eval()
            self.set_requires_grad([self.netF], False)


            self.netM = networks.define_FM().to(self.device)
            self.netM.eval()
            self.set_requires_grad([self.netM], False)


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
            

            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

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


            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B_M.parameters()), lr=opt.lr*1.5, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr*1, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_M)
            
            self.aa=np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
            self.bb=torch.autograd.variable(torch.from_numpy(self.aa).float().permute(0,3,1,2).cuda())
            self.FeaturesCalculated = False
            self.features = None



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

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            #out0, out1, out2, out3 = self.calculate_Features(self.upsample_Feature(self.real_B))
            if(not self.FeaturesCalculated):
                self.features = self.netM(self.model)
                self.FeaturesCalculated = True

            
            
      

            input_net = torch.cat([self.real_B, self.real_B_depth], 1)
            self.fake_A, selectedModels, self.out_depth = self.netG_B(input_net, self.features)
            selectedModels = selectedModels.squeeze(0)
            selectedModels = selectedModels.squeeze(1)
            selectedModels = selectedModels.squeeze(1)
            selectedModels = selectedModels.cpu().detach().numpy()

            #print(selectedModels)
            ind = np.argpartition(selectedModels, -4)[-4:]
            #print(ind)

            #index0 = ind[0]
            #index1 = ind[1]

            #print("index0", index0, "index1", index1, "prob:", prob.cpu().detach().numpy()[0])

            self.model0 = self.model[ind[0], :, :, :].unsqueeze(0)
            self.model1 = self.model[ind[1], :, :, :].unsqueeze(0)
            self.model2 = self.model[ind[2], :, :, :].unsqueeze(0)
            self.model3 = self.model[ind[3], :, :, :].unsqueeze(0)  

        else:
            self.fake_A = self.netG_B(self.real_B)


    def backward_D_basic(self, netD, real, fake, real_mask, fake_mask):


        real_mask = self.upsample_M(real_mask)
        pred_real, squeeze_real = netD(real)
        pred_real = self.upsample_M(pred_real)
        squeeze_real = self.upsample_M(squeeze_real) * real_mask
        loss_D_real = self.criterionGAN(pred_real, True, real_mask)

        fake_mask = self.upsample_M(fake_mask.unsqueeze(0))
        pred_fake, squeeze_fake = netD(fake.detach())
        pred_fake = self.upsample_M(pred_fake)
        squeeze_fake = self.upsample_M(squeeze_fake) * fake_mask
        loss_D_fake = self.criterionGAN(pred_fake, False, fake_mask)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #print(loss_D)
        return loss_D, self.lay1(squeeze_real[:,0:3, :]), self.lay1(squeeze_fake[:,0:3, :])

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        
        image_mask = self.fake_A_pool.query(torch.cat([self.fake_A, self.real_B_mask], 1))
        fake_A = image_mask[:,0:3,:,:]
        real_B_mask = image_mask[:,3,:,:]
        PIL_fake_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(fake_A)))
        np_fake_A = np.array(PIL_fake_A_Jitter)


        #if(random.random() > 0.5 and self.opt.lambda_identity>0):
            #PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.real_D)))
        #else:
        PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.real_A)))
        np_real_A = np.array(PIL_real_A_Jitter)


        np_real_B = np.array(self.jitter(Image.fromarray(util.tensor2im(self.real_B))))
        if(random.random() > 2):
            colorTransfer_real_A = color_transfer(np_real_B, np_real_A, clip=True, preserve_paper=False)
            #print(colorTransfer_real_A)
        else:
            colorTransfer_real_A = np_real_A

        self.real_A_D = util.im2tensor(np.array(util.tensor2im(colorTransfer_real_A)))
        self.fake_A_D = util.im2tensor(np_fake_A)

        self.real_A_D_S = self.real_A_D * self.real_A_mask
        self.fake_A_D_S = self.fake_A_D * real_B_mask

        toVGG = torch.cat([self.fake_A_D, self.real_A_D], 0)
        out0, out1, out2, out3 = self.calculate_Features(self.upsample_Feature(toVGG))

        concat_fake_A_M = torch.cat([ self.upsample_M(out0[0,:,:,:].unsqueeze(0)), self.upsample_M(out1[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_M = torch.cat([self.upsample_M(fake_A), concat_fake_A_M], 1)

        concat_real_A_M = torch.cat([ self.upsample_M(out0[1,:,:,:].unsqueeze(0)), self.upsample_M(out1[1,:,:,:].unsqueeze(0)), self.upsample_M(out2[1,:,:,:].unsqueeze(0))], 1)






        self.loss_D_B_M, self.squeeze_real, self.squeeze_fake = self.backward_D_basic(self.netD_B_M, concat_real_A_M, concat_fake_A_M, self.real_A_mask, real_B_mask)
        #self.loss_D_B_G, _, _ = self.backward_D_basic(self.netD_B_G, concat_real_A_G, concat_fake_A_G)


    def calculate_Features(self, image):
        
        generated = (image+1.0)/2.0*255.0
        input_image = generated-self.bb
        return self.netF(input_image)

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.opt.lambda_identity
        #if(self.opt.lambda_identity>0):
            #self.idt = self.netG_B(self.real_D)
            #self.loss_idt_A = self.criterionL1(self.idt, self.real_D)
            #self.loss_idt_A = lambda_idt * self.loss_idt_A
        #else:
        self.loss_idt_A = 0
        

        toVGG = torch.cat([self.fake_A, self.real_B], 0)
	
        out0, out1, out2, out3 = self.calculate_Features(self.upsample_Feature(toVGG))
        self.loss_F_B_F, _, _, _= self.criterionFeatureRobots(out0[1,:,:,:], out1[1,:,:,:], out2[1,:,:,:], out0[0,:,:,:], out1[0,:,:,:], out2[0,:,:,:])

        backgroundMask = ~(self.real_B_mask>0)
        self.fake_A_background = self.fake_A * backgroundMask
        self.real_B_background = self.real_B * backgroundMask


        #self.loss_F_B_F = self.criterionL1(self.fake_A, self.real_B)*0.2
        self.loss_F_B_B = self.criterionL1(self.fake_A_background, self.real_B_background)*2

        self.loss_F_B = self.loss_F_B_B + self.loss_F_B_F

        concat_fake_A_M = torch.cat([ self.upsample_M(out0[0,:,:,:].unsqueeze(0)), self.upsample_M(out1[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)



        b, squeeze =self.netD_B_M(concat_fake_A_M)

        #self.loss_G_B_L = self.criterionGAN(a, True)
        #mult = torch.autograd.variable(self.disc_div/torch.sum(self.real_B_mask))
        
        self.loss_G_B = self.criterionGAN(self.upsample_M(b), True, self.upsample_M(self.real_B_mask))
        #print("loss_G_B_M",self.loss_G_B_M)
        #self.loss_G_B_G = self.criterionGAN(c, True)
        #self.loss_G_B_M = self.loss_G_B_M# * mult
        #print("loss_G_B_M2", self.loss_G_B_M)
        #self.loss_G_B = self.loss_G_B_M# + self.loss_G_B_G * 1/3

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

        #self.optimizer_M.step()

        
    def get_one_hot(self,I, labels):
        I = I * 2.0 / 255.0 - 1
        tensor = (torch.abs(labels[0,:,:] - I)<0.001)
        return tensor


    def one_hot(self, image, labels):
        labels=labels.squeeze(0)
        image=image.squeeze(0)
        channels = []

        channels.append(self.get_one_hot(0,labels) )#background
        channels.append(self.get_one_hot(51,labels) )#shirt
        channels.append(self.get_one_hot(101,labels) )#robots
        channels.append(self.get_one_hot(151,labels) )#lines
        channels.append(self.get_one_hot(255,labels) )#field

        for channel in channels:
            image = torch.cat([image,channel.unsqueeze(0).float()],0)
        image.unsqueeze(0)
        
        return image


