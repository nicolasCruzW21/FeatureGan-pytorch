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

        self.loss_names = ['D_B_G', 'D_B_M', 'G_B','G_B_M','G_B_G', 'F_B', 'F_B_Image', 'F_B_Extra', 'G', 'idt_B','idt_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []
        visual_names_B = ['fake_A','real_B']

        if self.isTrain:
            visual_names_A = []
            visual_names_B = ['real_A','real_B','real_L','extra_real_B', 'fake_A', 'fake_A_D','squeeze_fake_M1','squeeze_fake_M2','squeeze_fake_G1','squeeze_fake_G2']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_B', 'D_B_G', 'D_B_M']
        else:  # during test time, only load Gs
            self.model_names = ['G_B']

        self.netG_B = networks.define_G(22, 3, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'cascade', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            #self.netD_A = networks.define_D(opt.output_nc, opt.ndf, "pixel",
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            #self.netD_B = networks.define_D(387, 72, "n_layers", 
                                            #2, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 2, True)

            self.netD_B_M = networks.define_D(387, 72, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 0, True)

            self.netD_B_G= networks.define_D(387, 72, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, 0, True)
            #self.netF = torch.jit.load("VGG19.pt") #load the BGG 19 network (feature extractor network)
            #self.netF=self.netF.eval()
            #self.set_requires_grad([self.netF], False)

            self.netF = networks.define_F(self.gpu_ids)
            self.netF=self.netF.eval()
            self.set_requires_grad([self.netF], False)
 
        self.lay0 = torch.nn.InstanceNorm2d(3, affine=False).cuda()


        self.layInst2 = torch.nn.InstanceNorm2d(6, affine=False).cuda()




        self.lay1 = torch.nn.LayerNorm([3, 256, 256], elementwise_affine=False).cuda()
        self.upsample_Feature = torch.nn.Upsample(size=256, mode='bilinear').cuda()
        self.criterionIdt = torch.nn.L1Loss()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

           
            factorImage = 3#400000 #4 zebra 10 elephant #0.5 last of us
            factorSky = 0.5
                
            
            self.criterionFeatureImage = networks.FeatureLoss(2*factorImage ,      2*factorImage,      1*factorImage,      "L1").to(self.device)
            self.criterionFeatureExtra = networks.FeatureLoss(2*factorSky ,      2*factorSky,      2*factorSky,      "L1").to(self.device)


            self.criterionL1 = torch.nn.L1Loss()
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

            self.avg_pool_disc = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

            #self.upsample_L = torch.nn.Upsample(size=opt.crop_size, mode='bilinear')
            self.upsample_M = torch.nn.Upsample(size=(int)(256), mode='bilinear')
            self.upsample_G = torch.nn.Upsample(size=(int)(128), mode='bilinear')
            self.upsample_Im = torch.nn.Upsample(size=(int)(256), mode='bilinear')

            

            self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.015, saturation=0.015, hue=0.05)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B_M.parameters(), self.netD_B_G.parameters()), lr=opt.lr*1.5, betas=(opt.beta1, 0.999))

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
            image = self.one_hot(self.real_B, self.real_L).unsqueeze(0)


            self.fake_A = self.netG_B(image)#self.avg_pool(self.real_B))# G_B(B)
            #print("self.fake_A",self.fake_A.size())
        else:
            image = self.one_hot(self.real_B, self.real_L).unsqueeze(0)
            self.fake_A = self.netG_B(image)#self.avg_pool(self.real_B))# G_B(B)


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

        out0, out1, out2, out3, _  = self.calculate_Features(self.upsample_Feature(toVGG), 'layer')

        


        #real_A=self.lay0(real_A)
        #fake_A=self.lay0(fake_A)    

        #concat_fake_A_L = torch.cat([self.upsample_L(out3[0,:,:,:].unsqueeze(0)), self.upsample_L(out2[0,:,:,:].unsqueeze(0))], 1)
        #concat_fake_A_L = torch.cat([self.upsample_L(fake_A), concat_fake_A_L], 1)

        #concat_real_A_L = torch.cat([self.upsample_L(out3[1,:,:,:].unsqueeze(0)), self.upsample_L(out2[1,:,:,:].unsqueeze(0))], 1)
        #concat_real_A_L = torch.cat([self.upsample_L(real_A), concat_real_A_L], 1)
        
    

        concat_fake_A_M = torch.cat([self.upsample_M(out3[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)
        concat_fake_A_M = torch.cat([self.upsample_M(fake_A), concat_fake_A_M], 1)

        concat_real_A_M = torch.cat([self.upsample_M(out3[1,:,:,:].unsqueeze(0)), self.upsample_M(out2[1,:,:,:].unsqueeze(0))], 1)
        concat_real_A_M = torch.cat([self.upsample_M(real_A), concat_real_A_M], 1)


        concat_fake_A_G = torch.cat([self.upsample_G(out3[0,:,:,:].unsqueeze(0)), self.upsample_G(out2[0,:,:,:].unsqueeze(0))], 1)
        concat_fake_A_G = torch.cat([self.upsample_G(fake_A), concat_fake_A_G], 1)

        concat_real_A_G = torch.cat([self.upsample_G(out3[1,:,:,:].unsqueeze(0)), self.upsample_G(out2[1,:,:,:].unsqueeze(0))], 1)
        concat_real_A_G = torch.cat([self.upsample_G(real_A), concat_real_A_G], 1)



        #self.loss_D_B,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B, concat_real_A_L, concat_fake_A_L)
        #self.squeeze_fake_L1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_fake_L2 = self.upsample_L(squeeze_fake[:,3:6, :])
        #self.squeeze_real_L1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_real_L2 = self.upsample_L(squeeze_fake[:,3:6, :])

        self.loss_D_B_M,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B_M, concat_real_A_M, concat_fake_A_M)
        self.squeeze_fake_M1 = self.upsample_M(squeeze_fake[:,0:3, :])
        self.squeeze_fake_M2 = self.upsample_M(squeeze_fake[:,3:6, :])
        #self.squeeze_real_M1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_real_M2 = self.upsample_L(squeeze_fake[:,3:6, :])

        self.loss_D_B_G,squeeze_real, squeeze_fake = self.backward_D_basic(self.netD_B_G, concat_real_A_G, concat_fake_A_G)
        self.squeeze_fake_G1 = self.upsample_M(squeeze_fake[:,0:3, :])
        self.squeeze_fake_G2 = self.upsample_M(squeeze_fake[:,3:6, :])
        #self.squeeze_real_G1 = self.upsample_L(squeeze_fake[:,0:3, :])
        #self.squeeze_real_G2 = self.upsample_L(squeeze_fake[:,3:6, :])


    def calculate_Features(self, image, normalize = 'none'):
        
        generated = (image+1.0)/2.0*255.0
        input_image = generated-self.bb
        if(normalize == 'instance'):
            input_image=(self.lay0(input_image)+1)*255.0
        elif(normalize == 'layer'):
            input_image=(self.lay1(input_image)+1)*255.0
 
        out0, out1, out2, out3, out4 =self.netF(input_image)
        return out0, out1, out2, out3, out4

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.opt.lambda_identity
        if(self.opt.lambda_identity>0):
            self.dist = self.upsample_M(self.real_A)
            self.idt_A = self.netG_B(self.dist)
            #print(self.idt_A.size(), self.dist.size())
            self.loss_idt_A = self.criterionL1(self.idt_A, self.real_A)
            self.loss_idt_A = lambda_idt * self.loss_idt_A
        else:
            self.loss_idt_A = 0
        
        bool_sky = self.get_RGB(70,130,180,self.real_L.squeeze(0)).unsqueeze(0)
        bool_person = self.get_RGB(220,20,60,self.real_L.squeeze(0)).unsqueeze(0)
        bool_rider = self.get_RGB(255,0,0,self.real_L.squeeze(0)).unsqueeze(0)
        bool_extra= bool_rider + bool_person + bool_sky
        self.extra_real_B = (self.real_B.squeeze(0) * bool_extra).unsqueeze(0)
        self.extra_fake_A = (self.fake_A.squeeze(0) * bool_extra).unsqueeze(0)



        toVGG0 = torch.cat([self.fake_A, self.real_B], 0)
        toVGG1 = torch.cat([self.extra_fake_A, self.extra_real_B], 0)
        toVGG = torch.cat([toVGG0, toVGG1], 0)

        #toVGG = self.upsample_Feature(toVGG)
        out0, out1, out2, out3, out4  = self.calculate_Features(self.upsample_Feature(toVGG), 'instance')

        

        self.loss_F_B_Image, _, _, _= self.criterionFeatureImage(out1[1,:,:,:], out3[1,:,:,:], out4[1,:,:,:], out1[0,:,:,:], out3[0,:,:,:], out4[0,:,:,:])
        self.loss_F_B_Image = max(0,(1-lambda_idt)) * self.loss_F_B_Image
        self.loss_F_B_Extra, _, _, _= self.criterionFeatureExtra(out1[3,:,:,:], out3[3,:,:,:], out4[3,:,:,:], out1[2,:,:,:], out3[2,:,:,:], out4[2,:,:,:])

        concat_fake_A_M = torch.cat([self.upsample_M(out3[0,:,:,:].unsqueeze(0)), self.upsample_M(out2[0,:,:,:].unsqueeze(0))], 1)
        concat_fake_A_M = torch.cat([self.upsample_M(self.fake_A), concat_fake_A_M], 1)


        concat_fake_A_G = torch.cat([self.upsample_G(out3[0,:,:,:].unsqueeze(0)), self.upsample_G(out2[0,:,:,:].unsqueeze(0))], 1)
        concat_fake_A_G = torch.cat([self.upsample_G(self.fake_A), concat_fake_A_G], 1)

        #a, _ = self.netD_B(concat_fake_A_L)
        b, _ =self.netD_B_M(concat_fake_A_M)
        c, _ = self.netD_B_G(concat_fake_A_G)

        self.loss_G_B_L = 0#self.criterionGAN(a, True)
        self.loss_G_B_M = self.criterionGAN(b, True)
        self.loss_G_B_G = self.criterionGAN(c, True)

        self.loss_G_B = self.loss_G_B_M * 2/3 + self.loss_G_B_G * 1/3 #+ self.loss_G_B_L * 1/4

        self.loss_idt_B = 0



        

#--------------------------------Total------------------------------------------------------

        self.loss_F_B = self.loss_idt_B + self.loss_F_B_Image + self.loss_idt_A + self.loss_F_B_Extra
        # combined loss and calculate gradientsfake_A
        self.loss_G = self.loss_G_B + self.loss_F_B 
        #self.loss_G.backward()


    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        #self.set_requires_grad([self.netF], False)
        #with torch.cuda.amp.autocast(): 
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        

        self.set_requires_grad([self.netD_B_G, self.netD_B_M], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        
        #with torch.cuda.amp.autocast(): 
        self.backward_G(epoch)             # calculate gradients for G_A and G_B

        self.loss_G.backward()
        #self.scaler_G.step(self.optimizer_G) 
        #self.scaler_G.update()

        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_B_G, self.netD_B_M], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        #self.backward_D_A()      # calculate gradients for D_A

        #with torch.cuda.amp.autocast(): 
        self.backward_D_B()      # calculate graidents for D_B

        self.loss_D_B_G.backward()
        self.loss_D_B_M.backward()
        #self.loss_D_B.backward()
        
        self.optimizer_D.step()  # update D_A and D_B's weights
       #self.scaler_D.step(self.optimizer_D) 
        #self.scaler_D.update()
        #self.draw()
        
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

        channels.append(self.get_RGB(128,64,128,labels))#road
        channels.append(self.get_RGB(244,35,232,labels))#sidewalk
        channels.append(self.get_RGB(220,20,60,labels))#person
        channels.append(self.get_RGB(255,0,0,labels))#rider
        channels.append(self.get_RGB(0,0,142,labels))#car
        channels.append(self.get_RGB(0, 0, 70,labels))#truck
        channels.append(self.get_RGB(0, 60,100,labels))#bus
        channels.append(self.get_RGB(0, 80,100,labels))#train
        channels.append(self.get_RGB(0, 0,230,labels))#motorcycle
        channels.append(self.get_RGB(119, 11, 32,labels))#bicycle
        channels.append(self.get_RGB(70, 70, 70,labels))#building
        channels.append(self.get_RGB(102,102,156,labels))#wall
        channels.append(self.get_RGB(190,153,153,labels))#fence
        channels.append(self.get_RGB(153,153,153,labels))#pole
        channels.append(self.get_RGB(220,220, 0,labels))#trafic sign
        channels.append(self.get_RGB(250,170, 30,labels))#trafic light
        channels.append(self.get_RGB(107,142, 35,labels))#vegetation
        channels.append(self.get_RGB(152,251,152,labels))#terrain
        channels.append(self.get_RGB(70,130,180,labels))#sky
        
        
        for channel in channels:
            image = torch.cat([image,channel.unsqueeze(0)],0)
        image.unsqueeze(0)
        
        return image

    def draw(self):
        self.real_A = self.upsample_Im(self.real_A)
        self.real_B = self.upsample_Im(self.real_B)
        self.real_L = self.upsample_Im(self.real_L)
        self.fake_A = self.upsample_Im(self.fake_A)
        self.fake_A_D = self.upsample_Im(self.fake_A_D)
        self.extra_real_B = self.upsample_Im(self.extra_real_B)
        self.squeeze_fake_M1 = self.upsample_Im(self.squeeze_fake_M1)
        self.squeeze_fake_M2 = self.upsample_Im(self.squeeze_fake_M2)
        self.squeeze_fake_G1 = self.upsample_Im(self.squeeze_fake_G1)
        self.squeeze_fake_G2 = self.upsample_Im(self.squeeze_fake_G2)

        

