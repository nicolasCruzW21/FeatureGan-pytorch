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

#TODO: cambiar la textura del piso del field
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        
        Parameters:
        """
        BaseModel.__init__(self, opt)

        #self.scaler_G = torch.cuda.amp.GradScaler() 
        #self.scaler_D = torch.cuda.amp.GradScaler() 

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_mean_M_real = 50
        self.loss_var_M_real = 90
        
        self.loss_mean_M_fake = 0
        self.loss_var_M_fake = 0

        self.loss_mean_M=0
        self.loss_var_M=0



        self.loss_mean_G_real = 50
        self.loss_var_G_real = 90

        self.loss_mean_G_fake = 0
        self.loss_var_G_fake = 0

        self.loss_mean_G=0
        self.loss_var_G=0

        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []

        if self.isTrain:
            visual_names_A = []
            visual_names_B = ['real_A','real_B', 'fake_A', 'fake_A_D','squeeze_real_M1','squeeze_real_G1','squeeze_fake_M1','squeeze_fake_G1']
            self.loss_names = ['D_B_G', 'D_B_M', 'G_B','G_B_M', 'G_B_G', 'F_B', 'F_B_Image', 'F_B_Small', 'F_B_Big', 'G']
        else:
            visual_names_B = ['gtFine_labelIds', 'leftImg8bit', 'leftImg8bit_r']
            self.loss_names = ['idt_B']
        

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_B', 'D_B_G', 'D_B_M']
        else:  # during test time, only load Gs
            self.model_names = ['G_B']

        self.netG_B = networks.define_G(32, 32, 72, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)



        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'cascade', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            #self.netD_A = networks.define_D(opt.output_nc, opt.ndf, "pixel",
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            #self.netD_B_L = networks.define_D(387, 72, "n_layers", 
                                            #2, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 2, True)

            self.netD_B_M = networks.define_D(320, 64, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 1, False, True)

            self.netD_B_G= networks.define_D(896, 64, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 1, False, True)
            #self.netF = torch.jit.load("VGG19.pt") #load the BGG 19 network (feature extractor network)
            #self.netF=self.netF.eval()
            #self.set_requires_grad([self.netF], False)

            self.netF = self.netF = torch.jit.load("VGG19.pt")#networks.define_F(self.gpu_ids)
            self.netF=self.netF.eval()
            self.set_requires_grad([self.netF], False)

            


        else:
            self.netG_B=self.netG_B.eval()
            self.set_requires_grad([self.netG_B], False)
		
 
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

           
            factorImage = 2#400000 #4 zebra 10 elephant #0.5 last of us
            factorSmall = 5
            factorBIg = 2
                
            
            self.criterionFeatureImage = networks.FeatureLoss(4*factorImage ,      4*factorImage,      1*factorImage,      "l1").to(self.device)
            self.criterionFeatureSmall = networks.FeatureLoss(2*factorSmall ,      2*factorSmall,      1*factorSmall,      "MSE").to(self.device)
            self.criterionFeatureBig = networks.FeatureLoss(1*factorBIg ,      2*factorBIg,      1*factorBIg,      "L1").to(self.device)


            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            

            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

            self.avg_pool_disc = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

            self.upsample_L = torch.nn.Upsample(size=opt.crop_size, mode='nearest')
            self.upsample_L_seg = torch.nn.Upsample(size=opt.crop_size, mode='nearest')

            self.upsample_M = torch.nn.Upsample(size=(int)(256), mode='nearest')
            self.upsample_G = torch.nn.Upsample(size=(int)(128), mode='nearest')
            self.upsample_crop = torch.nn.Upsample(size=(int)(opt.crop_size), mode='nearest')

            

            self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.02)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B_M.parameters(), self.netD_B_G.parameters()), lr=opt.lr*2, betas=(opt.beta1, 0.999))

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
        self.real_L = input['L' if AtoB else 'L'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            #with torch.cuda.amp.autocast():
            #print("self.real_B",self.real_B.size())
            self.hot_image = self.one_hot(self.real_B, self.real_L).unsqueeze(0)


            result = self.netG_B(self.hot_image)#self.avg_pool(self.real_B))# G_B(B)
            self.fake_A = result[:,:3,:,:]
            #self.seg = result[:,3:,:,:]


            #result = self.netG_B(torch.cat([self.real_A,self.hot_image[:,3:,:,:]*0],1))#self.avg_pool(self.real_B))# G_B(B)
            #self.idt = result[:,:3,:,:]
            #self.seg_idt = self.sig(result[:,3:,:,:])
            #print("self.fake_A",self.fake_A.size())
        else:
            self.gtFine_labelIds = self.getIds(self.real_L).unsqueeze(0)
            image = self.one_hot(self.real_B, self.real_L).unsqueeze(0)
            result = self.netG_B(image)
            self.leftImg8bit =result[:,:3,:,:]
            self.leftImg8bit_r = self.real_B
            self.loss_idt_B = 0#self.criterionSeg(result[:,3:,:,:],image[:,3:,:,:])
                
            #self.seg_cars = self.sig(result[:,21:24,:,:])


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        #with torch.cuda.amp.autocast():
        pred_real, squeeze_real = netD(real)
        
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        #with torch.cuda.amp.autocast():
        pred_fake, squeeze_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        #print('---------------------', self.squeeze_fake_1.size())
        
        # Combined loss and calculate gradients
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #loss_D.backward()
        return loss_D, squeeze_real, squeeze_fake

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)

        self.fake_A_D = fake_A
        PIL_fake_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(fake_A)))
        fake_A = util.im2tensor(np.array(PIL_fake_A_Jitter))


        PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.real_A)))

        real_A = util.im2tensor(np.array(PIL_real_A_Jitter))
        

        toVGG = torch.cat([fake_A, real_A], 0)

        out0, out1, out2, out3, out4 = self.calculate_Features(self.upsample_Feature(toVGG))

        


        #real_A=self.lay0(real_A)
        #fake_A=self.lay0(fake_A)    

        #concat_fake_A_L = torch.cat([self.upsample_L(out3[0,:,:,:].unsqueeze(0)), self.upsample_L(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_L = torch.cat([self.upsample_L(fake_A), concat_fake_A_L], 1)

        #concat_real_A_L = torch.cat([self.upsample_L(out3[1,:,:,:].unsqueeze(0)), self.upsample_L(out2[1,:,:,:].unsqueeze(0))], 1)
        #concat_real_A_L = torch.cat([self.upsample_L(real_A), concat_real_A_L], 1)
        
    

        concat_fake_A_M = torch.cat([ self.upsample_M(out0[0,:,:,:].unsqueeze(0)), self.upsample_M(out1[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_M = torch.cat([self.upsample_M(fake_A), concat_fake_A_M], 1)

        concat_real_A_M = torch.cat([ self.upsample_M(out0[1,:,:,:].unsqueeze(0)), self.upsample_M(out1[1,:,:,:].unsqueeze(0)), self.upsample_M(out2[1,:,:,:].unsqueeze(0))], 1)
        #concat_real_A_M = torch.cat([self.upsample_M(real_A), concat_real_A_M], 1)
        #self.loss_mean_M_real = 0.0001* torch.mean(concat_real_A_M).detach().requires_grad_(True) + 0.9999*self.loss_mean_M_real
        #self.loss_var_M_real = 0.0001* math.sqrt(torch.var(concat_real_A_M).detach().requires_grad_(True)) + 0.9999*self.loss_var_M_real

        concat_fake_A_G = torch.cat([ self.upsample_G(out2[0,:,:,:].unsqueeze(0)), self.upsample_G(out3[0,:,:,:].unsqueeze(0)), self.upsample_G(out4[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_G = torch.cat([self.upsample_G(fake_A), concat_fake_A_G], 1)

        concat_real_A_G = torch.cat([ self.upsample_G(out2[1,:,:,:].unsqueeze(0)), self.upsample_G(out3[1,:,:,:].unsqueeze(0)), self.upsample_G(out4[1,:,:,:].unsqueeze(0))], 1)
        #concat_real_A_G = torch.cat([self.upsample_G(real_A), concat_real_A_G], 1)
        #self.loss_mean_G_real = 0.0001* torch.mean(concat_real_A_G).detach().requires_grad_(True) + 0.9999*self.loss_mean_G_real
        #self.loss_var_G_real = 0.0001* math.sqrt(torch.var(concat_real_A_G).detach().requires_grad_(True)) + 0.9999*self.loss_var_G_real




        #self.loss_D_B_L,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B_L, concat_real_A_L, concat_fake_A_L)
        #self.squeeze_fake_L1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_fake_L2 = self.upsample_L(squeeze_fake[:,3:6, :])
        #self.squeeze_real_L1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_real_L2 = self.upsample_L(squeeze_fake[:,3:6, :])

        self.loss_D_B_M,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B_M, concat_real_A_M, concat_fake_A_M)
        self.squeeze_fake_M1 = self.lay1(self.upsample_crop(squeeze_fake[:,0:3, :]))
        #self.squeeze_fake_M2 = self.upsample_M(squeeze_fake[:,3:6, :])
        self.squeeze_real_M1 = self.lay1(self.upsample_crop(squeeze_real[:,0:3, :]))
        #self.squeeze_real_M2 = self.upsample_M(squeeze_fake[:,3:6, :])

        self.loss_D_B_G,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B_G, concat_real_A_G, concat_fake_A_G)
        self.squeeze_fake_G1 = self.lay1(self.upsample_crop(squeeze_fake[:,0:3, :]))
        #self.squeeze_fake_G2 = self.upsample_M(squeeze_fake[:,3:6, :])
        self.squeeze_real_G1 = self.lay1(self.upsample_crop(squeeze_real[:,0:3, :]))
        #self.squeeze_real_G2 = self.upsample_M(squeeze_fake[:,3:6, :])


    def calculate_Features(self, image):
        
        generated = (image+1.0)/2.0*255.0
        input_image = generated-self.bb
        #traced_script_module = torch.jit.trace(self.netF, input_image)
        #traced_script_module.save("VGG19.pt")
        #assert(False)
        return self.netF(input_image)

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.opt.lambda_identity
        if(self.opt.lambda_identity>0):

            self.loss_idt_A = self.criterionL1(self.idt, self.real_A)
            self.loss_idt_A = lambda_idt * self.loss_idt_A
        else:
            self.loss_idt_A = 0
        
        
        bool_person = self.get_RGB(220,20,60,self.real_L.squeeze(0)).unsqueeze(0)
        bool_rider = self.get_RGB(255,0,0,self.real_L.squeeze(0)).unsqueeze(0)
        bool_pole = self.get_RGB(153,153,153,self.real_L.squeeze(0)).unsqueeze(0)
        bool_trafic_light = self.get_RGB(250,170, 30,self.real_L.squeeze(0)).unsqueeze(0)
        bool_trafic_sign = self.get_RGB(220,220, 0,self.real_L.squeeze(0)).unsqueeze(0)
        bool_motorcycle = self.get_RGB(0, 0,230,self.real_L.squeeze(0)).unsqueeze(0)
        bool_bicycle = self.get_RGB(119, 11, 32,self.real_L.squeeze(0)).unsqueeze(0)

        bool_car = self.get_RGB(0,0,142, self.real_L.squeeze(0)).unsqueeze(0)
        bool_truck = self.get_RGB(0, 0, 70, self.real_L.squeeze(0)).unsqueeze(0)
        bool_bus = self.get_RGB(0, 60,100, self.real_L.squeeze(0)).unsqueeze(0)
        bool_caravan = self.get_RGB(0, 0, 90, self.real_L.squeeze(0)).unsqueeze(0)
        bool_trailer = self.get_RGB(0, 0, 110,self.real_L.squeeze(0)).unsqueeze(0)
        bool_train = self.get_RGB(0, 80,100, self.real_L.squeeze(0)).unsqueeze(0)
        bool_vegetation = self.get_RGB(107,142, 35, self.real_L.squeeze(0)).unsqueeze(0)
        bool_sky = self.get_RGB(70,130,180,self.real_L.squeeze(0)).unsqueeze(0)  
        bool_bridge = self.get_RGB(150, 100, 100,self.real_L.squeeze(0)).unsqueeze(0)
        bool_building = self.get_RGB(70, 70, 70,self.real_L.squeeze(0)).unsqueeze(0)
        bool_wall = self.get_RGB(102,102,156,self.real_L.squeeze(0)).unsqueeze(0)

        bool_bridge = self.get_RGB(150, 100, 100,self.real_L.squeeze(0)).unsqueeze(0)   
        bool_tunnel = self.get_RGB(150, 120, 90,self.real_L.squeeze(0)).unsqueeze(0)
        bool_terrain = self.get_RGB(152,251,152,self.real_L.squeeze(0)).unsqueeze(0)
        bool_sidewalk = self.get_RGB(244,35,232,self.real_L.squeeze(0)).unsqueeze(0)
        bool_road = self.get_RGB(128,64,128,self.real_L.squeeze(0)).unsqueeze(0)



        bool_unlabeled = self.get_RGB(0,0,0,self.real_L.squeeze(0)).unsqueeze(0) #unlabeled
        bool_dynamic = self.get_RGB(111, 74, 0,self.real_L.squeeze(0)).unsqueeze(0)
        bool_ground = self.get_RGB(81, 0, 81,self.real_L.squeeze(0)).unsqueeze(0)
        bool_parking = self.get_RGB(250, 170, 160,self.real_L.squeeze(0)).unsqueeze(0)
        bool_rail_track = self.get_RGB(230, 150, 140,self.real_L.squeeze(0)).unsqueeze(0)
        bool_guard_rail = self.get_RGB(180, 165, 180,self.real_L.squeeze(0)).unsqueeze(0)
        bool_bridge = self.get_RGB(150, 100, 100,self.real_L.squeeze(0)).unsqueeze(0)
        bool_tunnel = self.get_RGB(150, 120, 90,self.real_L.squeeze(0)).unsqueeze(0)
        bool_caravan = self.get_RGB(0, 0, 90,self.real_L.squeeze(0)).unsqueeze(0)
        bool_trailer = self.get_RGB(0, 0, 110,self.real_L.squeeze(0)).unsqueeze(0)
        bool_license = self.get_RGB(0, 0, 142,self.real_L.squeeze(0)).unsqueeze(0)

        bool_ignore = bool_unlabeled + bool_dynamic + bool_ground + bool_parking + bool_rail_track + bool_guard_rail + bool_bridge + bool_tunnel + bool_caravan + bool_trailer + bool_terrain + bool_sky
        bool_check = ~bool_ignore





        self.small_real_B = (self.real_B.squeeze(0) * bool_check).unsqueeze(0)
        self.small_fake_A = (self.fake_A.squeeze(0) * bool_check).unsqueeze(0)


        






        bool_big = bool_terrain + bool_sky + bool_pole + bool_trafic_light + bool_trafic_sign
        self.big_real_B = (self.real_B.squeeze(0) * bool_big).unsqueeze(0)
        self.big_fake_A = (self.fake_A.squeeze(0) * bool_big).unsqueeze(0)



        toVGG0 = torch.cat([self.fake_A, self.real_B], 0)
        #toVGG1 = torch.cat([self.small_fake_A, self.small_real_B], 0)
        #toVGG2 = torch.cat([self.big_fake_A, self.big_real_B], 0)
        #toVGG = torch.cat([toVGG0, toVGG1, toVGG2], 0)

        #Feature Loss
        out0, out1, out2, out3, out4  = self.calculate_Features(self.upsample_Feature(toVGG0))
        self.loss_F_B_Image , _, _, _= self.criterionFeatureImage(out1[1,:,:,:], out2[1,:,:,:], out4[1,:,:,:], out1[0,:,:,:], out2[0,:,:,:], out4[0,:,:,:])
        self.loss_F_B_Small = 0#, _, _, _= self.criterionFeatureSmall(out1[3,:,:,:], out2[3,:,:,:], out4[3,:,:,:], out1[2,:,:,:], out2[2,:,:,:], out4[2,:,:,:])
        self.loss_F_B_Big= 0#, _, _, _= 0#self.criterionFeatureBig(out0[5,:,:,:], out2[5,:,:,:], out3[5,:,:,:], out0[4,:,:,:], out2[4,:,:,:], out3[4,:,:,:])

        #concat_fake_A_L = torch.cat([self.upsample_L(out3[0,:,:,:].unsqueeze(0)), self.upsample_L(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_L = torch.cat([self.upsample_L(self.fake_A), concat_fake_A_L], 1)

        concat_fake_A_M = torch.cat([ self.upsample_M(out0[0,:,:,:].unsqueeze(0)), self.upsample_M(out1[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_M = torch.cat([self.upsample_M(self.fake_A), concat_fake_A_M], 1)

        #self.loss_gauss_M = 1-self.normpdf(torch.mean(concat_fake_A_M).detach().requires_grad_(True), self.loss_mean_M_real, 5)


        concat_fake_A_G = torch.cat([ self.upsample_G(out2[0,:,:,:].unsqueeze(0)), self.upsample_G(out3[0,:,:,:].unsqueeze(0)), self.upsample_G(out4[0,:,:,:].unsqueeze(0))], 1)

        #self.loss_gauss_G = 1-self.normpdf(torch.mean(concat_fake_A_G).detach().requires_grad_(True), self.loss_mean_M_real, 5)


        

        #a, _ = self.netD_B_L(concat_fake_A_L)
        b, _ =self.netD_B_M(concat_fake_A_M)
        c, _ = self.netD_B_G(concat_fake_A_G)

        #self.loss_G_B_L = self.criterionGAN(a, True)
        self.loss_G_B_M = self.criterionGAN(b, True)
        self.loss_G_B_G = self.criterionGAN(c, True)

        self.loss_G_B = self.loss_G_B_M * 1/2 + self.loss_G_B_G * 1/2 #+ self.loss_G_B_L * 1/4
        
        self.loss_idt_B =  0#3*self.criterionSeg(self.seg, self.hot_image[:,3:,:,:])
        
        #self.seg_cars = self.sig(self.seg[:,18:21,:,:])
        #self.seg_cars_idt = self.seg_idt[:,18:21,:,:]
        #self.seg_cars_target = self.hot_image[:,21:24,:,:]


        

#--------------------------------Total------------------------------------------------------


        self.loss_F_B = self.loss_F_B_Image + self.loss_F_B_Small + self.loss_F_B_Big
        # combined loss and calculate gradientsfake_A
        self.loss_G = self.loss_F_B + self.loss_idt_B + self.loss_G_B   
        #self.loss_G.backward()

    def normpdf(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        #self.set_requires_grad([self.netF], False)
        #with torch.cuda.amp.autocast(): 
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        

        # D_A and D_B
        self.set_requires_grad([self.netD_B_G, self.netD_B_M], True)
        self.set_requires_grad([self.netG_B], False)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_B()      # calculate graidents for D_B

        self.loss_D_B_G.backward()
        self.loss_D_B_M.backward()
        #self.loss_D_B_L.backward(retain_graph=True)
        self.optimizer_D.step()  # update D_A and D_B's weights


        self.set_requires_grad([self.netD_B_G, self.netD_B_M], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_B], True)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        
        #with torch.cuda.amp.autocast(): 
        self.backward_G(epoch)             # calculate gradients for G_A and G_B

        self.loss_G.backward(retain_graph=True)
        
        #self.scaler_G.step(self.optimizer_G) 
        #self.scaler_G.update()

        self.optimizer_G.step()       # update G_A and G_B's weights

        
    def get_RGB(self,R,G,B, labels):
        #print("--------------------",labels.size())
        R = R * 2.0 / 255.0 - 1
        G = G * 2.0 / 255.0 - 1
        B = B * 2.0 / 255.0 - 1
        tensor = (torch.abs(labels[0,:,:] - R)<0.00001) * (torch.abs(labels[1,:,:] - G)<0.00001) * (torch.abs(labels[2,:,:] - B)<0.00001)
        return tensor


    def one_hot(self, image, labels):
        labels=labels.squeeze(0)
        image=image.squeeze(0)
        channels = []

        channels.append(self.get_RGB(0,  0,  0,labels) )#ego vehicle 0
        channels.append(self.get_RGB(111, 74,  0,labels))#dynamic 1
        channels.append(self.get_RGB(81,  0, 81,labels))#ground 2
        channels.append(self.get_RGB(128,64,128,labels) )#road 3
        channels.append(self.get_RGB(244,35,232,labels) )#sidewalk 4
        channels.append(self.get_RGB(250,170,160,labels) )#building 5
        channels.append(self.get_RGB(230,150,140,labels) )#building 6
        channels.append(self.get_RGB(70, 70, 70,labels) )#building 7
        channels.append(self.get_RGB(102,102,156,labels) )#wall 8
        channels.append(self.get_RGB(190,153,153,labels) )#fence 9
        channels.append(self.get_RGB(180, 165, 180,labels) )#guard rail 10
        channels.append(self.get_RGB(150, 100, 100,labels) )#bridge 11
        channels.append(self.get_RGB(150, 120, 90,labels) )#tunnel 12
        channels.append(self.get_RGB(153,153,153,labels) )#polegroup 13
        channels.append(self.get_RGB(250,170, 30,labels) )#trafic light 14
        channels.append(self.get_RGB(220,220, 0,labels) )#trafic sign 15
        channels.append(self.get_RGB(107,142, 35,labels) )#vegetation 16
        channels.append(self.get_RGB(152,251,152,labels) )#terrain 17
        channels.append(self.get_RGB(70,130,180,labels) )#sky 18
        channels.append(self.get_RGB(220,20,60,labels) )#person 19
        channels.append(self.get_RGB(255,0,0,labels) )#rider 20
        channels.append(self.get_RGB(0,0,142,labels) )#car 21
        channels.append(self.get_RGB(0, 0, 70,labels) )#truck 22
        channels.append(self.get_RGB(0, 60,100,labels) )#bus 23
        channels.append(self.get_RGB(0, 0, 90,labels) )#caravan 24
        channels.append(self.get_RGB(0, 0, 110,labels) )#trailer 25
        channels.append(self.get_RGB(0, 80,100,labels) )#train 26
        channels.append(self.get_RGB(0, 0,230,labels) )#motorcycle 27
        channels.append(self.get_RGB(119, 11, 32,labels) )#bicycle 28


        
        
        for channel in channels:
            image = torch.cat([image,channel.unsqueeze(0).float()],0)
        image.unsqueeze(0)
        
        return image


    def getIds(self, labels):
        labels=labels.squeeze(0)
        channels = []
        #channels.append(self.get_RGB(0,  0,  0,labels) * 0)#unlabeled
        channels.append(self.get_RGB(0,  0,  0,labels) * 0)#ego vehicle
        #channels.append(self.get_RGB(0,  0,  0,labels) * 2)#rectification border
        #channels.append(self.get_RGB(0,  0,  0,labels) * 3)#out of roi
        #channels.append(self.get_RGB(0,  0,  0,labels) * 4)#static
        channels.append(self.get_RGB(111, 74,  0,labels) * 5)#dynamic
        channels.append(self.get_RGB(81,  0, 81,labels) * 6)#ground
        channels.append(self.get_RGB(128,64,128,labels) * 7)#road
        channels.append(self.get_RGB(244,35,232,labels) * 8)#sidewalk
        channels.append(self.get_RGB(250,170,160,labels) * 9)#parking
        channels.append(self.get_RGB(230,150,140,labels) * 10)#rail track
        channels.append(self.get_RGB(70, 70, 70,labels) * 11)#building
        channels.append(self.get_RGB(102,102,156,labels) * 12)#wall
        channels.append(self.get_RGB(190,153,153,labels) * 13)#fence
        channels.append(self.get_RGB(180, 165, 180,labels) * 14)#guard rail
        channels.append(self.get_RGB(150, 100, 100,labels) * 15)#bridge
        channels.append(self.get_RGB(150, 120, 90,labels) * 16)#tunnel
        channels.append(self.get_RGB(153,153,153,labels) * 17)#pole
        channels.append(self.get_RGB(250,170, 30,labels) * 19)#trafic light
        channels.append(self.get_RGB(220,220, 0,labels) * 20)#trafic sign
        channels.append(self.get_RGB(107,142, 35,labels) * 21)#vegetation
        channels.append(self.get_RGB(152,251,152,labels) * 22)#terrain
        channels.append(self.get_RGB(70,130,180,labels) * 23)#sky
        channels.append(self.get_RGB(220,20,60,labels) * 24)#person
        channels.append(self.get_RGB(255,0,0,labels) * 25)#rider
        channels.append(self.get_RGB(0,0,142,labels) * 26)#car
        channels.append(self.get_RGB(0, 0, 70,labels) * 27)#truck
        channels.append(self.get_RGB(0, 60,100,labels) * 28)#bus
        channels.append(self.get_RGB(0, 0, 90,labels) * 29)#caravan
        channels.append(self.get_RGB(0, 0, 110,labels) * 30)#trailer
        channels.append(self.get_RGB(0, 80,100,labels) * 31)#train
        channels.append(self.get_RGB(0, 0,230,labels) * 32)#motorcycle
        channels.append(self.get_RGB(119, 11, 32,labels) * 33)#bicycle
        
        image=channels[0] * 0
        image = image.unsqueeze(0)
        for channel in channels:
            image = image + channel.unsqueeze(0)*1.0
        image = (image+1) * 2.0 / 255.0 - 1
        image.unsqueeze(0)
        return image

