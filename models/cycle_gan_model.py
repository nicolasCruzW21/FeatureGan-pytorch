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
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_B', 'G_B', 'G_B_L', 'G_B_G', 'D_real', 'D_fake' ,'D_G_real', 'D_G_fake' ,'F_B', 'F_B_field', 'F_B_RLG', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []
        visual_names_B = ['real_B', 'fake_A']
        if self.isTrain:
            visual_names_A = []
            visual_names_B = ['real_B', 'fake_A', 'norm_RGL_real_B', 'norm_RGL_fake_A', 'norm_field_real_B', 'norm_field_fake_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            print("------------idt is used-------------")
            #visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_B', 'D_B', 'D_B_G']
        else:  # during test time, only load Gs
            self.model_names = ['G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        #self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, 128, opt.netG, opt.norm,
                                        #not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc * 4, 256, opt.netG, opt.norm,
                                        #not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        #self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        #not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(6, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        self.ones = torch.ones([256,256]).cuda().float()
        self.ones3D = torch.ones([1, 256,256]).cuda().float()
        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'cascade', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            #self.netD_A = networks.define_D(opt.output_nc, opt.ndf, "pixel",
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_B_G = networks.define_D(opt.input_nc, opt.ndf, "n_layers", 
                                            5, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netF = networks.define_F(self.gpu_ids)
 

        self.backgroundFactor = 0.02
        self.foregroundFactor = 0.2 
        self.lay0 = torch.nn.InstanceNorm2d(3, affine=True).cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            
            factorField = 8
            factor = 16
                
            
            self.criterionFeatureField = networks.FeatureLoss(5.2*factorField , 10.8*factorField, 0.8*factorField).to(self.device)
            self.criterionFeature =      networks.FeatureLoss(5.2*factor ,      10.8*factor,      0.8*factor      ).to(self.device)
            #self.criterionFeature = networks.FeatureLoss(1000000 , 1000000, 1000000, 1).to(self.device)

            self.criterionCycle_A = torch.nn.L1Loss()
            self.criterionCycle_B = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bicubic')
            self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
            self.jitterIdt = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()), lr=opt.lr*2, betas=(opt.beta1, 0.999))

            self.optimizer_D_G = torch.optim.Adam(itertools.chain(self.netD_B_G.parameters()), lr=opt.lr*2, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_G)
            
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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:

            #PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.real_A)))
            #self.real_A = util.im2tensor(np.array(PIL_real_A_Jitter))
            #self.real_B, _ = self.remove_background(self.real_B, self.real_B, self.backgroundFactor)

            #self.fake_B = self.netG_A(self.real_A) # G_A(A)
            #self.rec_A_array = self.netG_B(self.fake_B)   # G_B(G_A(A)


            #lossList = []
            #for image in range (self.rec_A_array.size(0)):
                #lossList.insert(image, self.criterionCycle_A((self.rec_A_array[image,:]).unsqueeze(0), self.real_A))    
            #selectedImageIndex = lossList.index(min(lossList))

            #self.rec_A = (self.rec_A_array[selectedImageIndex,:]).unsqueeze(0)

            #self.rec_A_1 = (self.rec_A_array[1,:]).unsqueeze(0)
            #self.rec_A_2 = (self.rec_A_array[2,:]).unsqueeze(0)

            #self.fake_A_array = self.netG_B(self.add_background_foreground_channel(self.real_B, self.real_B))# G_B(B)
            self.real_B_channels = self.add_background_field_channel(self.real_B, self.real_B)
            self.fake_A_array = self.netG_B(self.real_B_channels)# G_B(B)


            self.fake_A = (self.fake_A_array[0,:]).unsqueeze(0)

            #self.smooth=nn.AvgPool2d(kernel_size=5, stride=1, padding = 2)
            #self.smooth_fake_A = self.smooth(self.fake_A)
            
            #PIL_fake_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.fake_A)))
            #fake_A_Jittered = util.im2tensor(np.array(PIL_fake_A_Jitter))
            #self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        else:
            #self.fake_B = self.netG_A(self.real_A)# G_A(A)


            #self.rec_A_array = self.netG_B(self.add_background_foreground_channel(self.fake_B, self.fake_B))# G_B(G_A(A))
            #self.rec_A_array = self.netG_B(self.fake_B)# G_B(G_A(A))
            #self.rec_A = (self.rec_A_array[0,:]).unsqueeze(0)

            #self.fake_A_array = self.netG_B(self.add_background_foreground_channel(self.real_B, self.real_B))# G_B(B)
            self.real_B_channels = self.add_background_field_channel(self.real_B, self.real_B)
            self.fake_A_array = self.netG_B(self.real_B_channels)# G_B(B)
            self.fake_A = (self.fake_A_array[0,:]).unsqueeze(0)

            #self.rec_B = self.netG_A(self.fake_A)# G_A(G_B(B))


    def backward_D_basic(self, netD, real, fake, Global):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Combined loss and calculate gradients
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        if(Global):
            self.loss_D_G_real = loss_D_real
            self.loss_D_G_fake = loss_D_fake
        else:
            self.loss_D_real = loss_D_real
            self.loss_D_fake = loss_D_fake
        return loss_D

    #def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        #self.loss_D_A_G = self.backward_D_basic(self.netD_A_G, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        PIL_fake_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(fake_A)))
        fake_A = util.im2tensor(np.array(PIL_fake_A_Jitter))


        PIL_real_A_Jitter = self.jitter(Image.fromarray(util.tensor2im(self.real_A)))
        real_A = util.im2tensor(np.array(PIL_real_A_Jitter))

        self.loss_D_B = self.backward_D_basic(self.netD_B, real_A, fake_A, False)
        self.loss_D_B_G = self.backward_D_basic(self.netD_B_G, real_A, fake_A, True)

    def remove_background(self, image, label, bound):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 50 * 2.0 / 255.0
        R = R - 1
        G = 50 * 2.0 / 255.0
        G = G - 1
        B = 50 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction < bound#0.02
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction < bound#0.02

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction < bound#0.02

        boolBack = booleanTenR * booleanTenG * booleanTenB
        #print("sum boolBack------------------",torch.sum(boolBack).cpu().float().detach().numpy())
        boolFront = ~boolBack
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())
        

        back = torch.cat(((self.ones*R).unsqueeze(0), (self.ones*G).unsqueeze(0), (self.ones*B).unsqueeze(0)),0)
        
        background = back * boolBack

        foreground = image * boolFront
        no_back_image = background + foreground
        no_back_image = no_back_image.unsqueeze(0)
        
        numBack = torch.sum(boolBack).cpu().float().detach().numpy()
        numFront = torch.sum(boolFront).cpu().float().detach().numpy()
        numFront = max(1, numFront)
        normFactor = (numFront+numBack) / numFront
        return no_back_image, normFactor

    def add_background(self, image, backgroundimage, label, bound):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 50 * 2.0 / 255.0
        R = R - 1
        G = 50 * 2.0 / 255.0
        G = G - 1
        B = 50 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction < bound#0.02
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction < bound#0.02

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction < bound#0.02

        boolBack = booleanTenR * booleanTenG * booleanTenB
        #print("sum boolBack------------------",torch.sum(boolBack).cpu().float().detach().numpy())
        boolFront = ~boolBack
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())
        

        back = torch.cat(((self.ones*R).unsqueeze(0), (self.ones*G).unsqueeze(0), (self.ones*B).unsqueeze(0)),0)
        
        background = backgroundimage * boolBack

        foreground = image * boolFront
        new_back_image = background + foreground
        new_back_image = new_back_image.unsqueeze(0)
        
        numBack = torch.sum(boolBack).cpu().float().detach().numpy()
        numFront = torch.sum(boolFront).cpu().float().detach().numpy()
        numFront = max(1, numFront)
        normFactor = (numFront+numBack) / numFront
        return new_back_image.squeeze(0), normFactor


    def add_background_field_channel(self, image, label):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 50 * 2.0 / 255.0
        R = R - 1
        G = 50 * 2.0 / 255.0
        G = G - 1
        B = 50 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction <self.backgroundFactor
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction <self.backgroundFactor

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction <self.backgroundFactor
        

        backGroundTen = booleanTenR * booleanTenG * booleanTenB
        #print("sum gT------------------",torch.sum(gT).cpu().float().detach().numpy())
        notbackGroundTen = ~backGroundTen
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())

        #foreground = image * notbackGroundTen
        #print("foreground",foreground.size())
        background = self.ones * backGroundTen
        #print("background",background.size())
        

        

#----------------------------------------field-----------------------------------------------


        R = 75.5 * 2.0 / 255.0
        R = R - 1
        G = 92.5 * 2.0 / 255.0
        G = G - 1
        B = 64 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction <self.foregroundFactor
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction <self.foregroundFactor

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction <self.foregroundFactor
        

        fieldTen = booleanTenR * booleanTenG * booleanTenB
        #print("sum fieldTen------------------",torch.sum(fieldTen).cpu().float().detach().numpy())
        notFieldTen = ~fieldTen
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())



        robots_goal_lines = self.ones * notFieldTen * notbackGroundTen
        #print("foreground",foreground.size())
        field = self.ones * fieldTen


        #print("background",background.size())
        
        finalImage = torch.cat((robots_goal_lines.unsqueeze(0), image),0)  

        finalImage = torch.cat((background.unsqueeze(0), finalImage),0)  
        #image_background_channel = background.unsqueeze(0)

        finalImage = torch.cat((field.unsqueeze(0), finalImage),0)
        
        finalImage = finalImage.unsqueeze(0)
        return finalImage

    def get_field_robot_line_goal_images(self, image, label):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 50 * 2.0 / 255.0
        R = R - 1
        G = 50 * 2.0 / 255.0
        G = G - 1
        B = 50 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction <self.backgroundFactor
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction <self.backgroundFactor

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction <self.backgroundFactor
        

        backGroundTen = booleanTenR * booleanTenG * booleanTenB
        #print("sum gT------------------",torch.sum(gT).cpu().float().detach().numpy())
        notbackGroundTen = ~backGroundTen
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())

        #foreground = image * notbackGroundTen
        #print("foreground",foreground.size())
        background = self.ones * backGroundTen
        #print("background",background.size())
        

        

#----------------------------------------field-----------------------------------------------


        R = 75.5 * 2.0 / 255.0
        R = R - 1
        G = 92.5 * 2.0 / 255.0
        G = G - 1
        B = 64 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction <self.foregroundFactor
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = torch.abs(channelG - aaa)
        booleanTenG = substraction <self.foregroundFactor

        #B
        aaa = self.ones*B
        channelB = label[2,:,:]
        substraction = torch.abs(channelB - aaa)
        booleanTenB = substraction <self.foregroundFactor
        

        fieldTen = booleanTenR * booleanTenG * booleanTenB
        #print("sum fieldTen------------------",torch.sum(fieldTen).cpu().float().detach().numpy())
        notFieldTen = ~fieldTen
        #print("sum ngT------------------",torch.sum(ngT).cpu().float().detach().numpy())

        robots_goal_lines = image * notFieldTen * notbackGroundTen
        robots_goal_lines = robots_goal_lines - self.ones * fieldTen - self.ones * backGroundTen
        #print("foreground",foreground.size())
        field = image * fieldTen
        field = field - self.ones * notFieldTen
        return robots_goal_lines.unsqueeze(0), field.unsqueeze(0)

    def calculate_Features(self, image):
        generated = (image+1.0)/2.0*255.0
        out7, out14, out23 =self.netF(generated-self.bb)
        return out7, out14, out23

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss


        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            PIL_real_B_Jitter = self.jitterIdt(Image.fromarray(util.tensor2im(self.real_B)))
            real_B_Jittered = util.im2tensor(np.array(PIL_real_B_Jitter))

            real_B_with_Background, _ = self.add_background(self.real_B, self.fake_A, self.real_B, self.backgroundFactor)

            self.idt_A = self.netG_A(real_B_with_Background)

            

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            #self.idt_B = self.netG_B(self.real_A)

            self.loss_idt_B = 0 #self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0



        #--------------------------------cycle A------------------------------------------------------

        #lossList = []
        #for image in range (self.rec_A_array.size(0)):
            #lossList.insert(image, self.criterionCycle_A((self.rec_A_array[image,:]).unsqueeze(0), self.real_A))
        self.loss_cycle_A = 0#(min(lossList)*0.99 + 0.01 * sum(lossList) / len(lossList)) * lambda_A


        #real_A_no_Background, normFactor = self.remove_background(self.real_A, self.fake_B, self.backgroundFactor)
        #fake_B_no_Background, normFactor = self.remove_background(self.fake_B, self.fake_B, self.backgroundFactor)
        #out7_r_A, out14_r_A, out23_r_A, out32_r_A  = self.calculate_Features(real_A_no_Background)
        #out7_f_B, out14_f_B, out23_f_A, out32_f_B  = self.calculate_Features(fake_B_no_Background)
        self.loss_F_A = 0#self.criterionFeature(out7_r_A, out14_r_A, out23_r_A, out32_r_A , out7_f_B, out14_f_B, out23_f_A, out32_f_B) * normFactor

        # GAN loss D_A(G_A(A))
        self.loss_G_A = 0#self.criterionGAN(self.netD_A(self.fake_B), True)*0.9 + self.criterionGAN(self.netD_A_G(self.fake_B), True)*0.1
        



        #--------------------------------cycle B------------------------------------------------------

        
        #print("foreground",foreground.size())
        #field = self.ones* fieldTen
        #field = self.real_B_channels[0,0,:,:] - self.ones * ~self.real_B_channels[0,0,:,:]

        robot_line_goal_real_B, field_real_B = self.get_field_robot_line_goal_images(self.real_B,self.real_B)
        robot_line_goal_fake_A, field_fake_A = self.get_field_robot_line_goal_images(self.fake_A,self.real_B)
        #print("test",test.size())

        #realBB = util.tensor2im((self.real_B_channels[0,2:5,:,:]).unsqueeze(0))

        #recBB = util.tensor2im((field_fake_A[0,0:3,:,:]).unsqueeze(0))


        #util.save_image(realBB, "realB.png")

        #util.save_image(recBB, "field.png")

        self.norm_RGL_fake_A = self.lay0(robot_line_goal_fake_A)
        self.norm_RGL_real_B = self.lay0(robot_line_goal_real_B)

        self.norm_field_fake_A = self.lay0(field_fake_A)
        self.norm_field_real_B = self.lay0(field_real_B)
        #print("norm_field_real_B", self.norm_field_real_B.size())


        #fake_A_no_Background, normFactor = self.remove_background(self.fake_A, self.real_B, self.backgroundFactor)
        #real_B_no_Background, normFactor = self.remove_background(self.real_B, self.real_B, self.backgroundFactor)
        out7_r_B, out14_r_B, out23_r_B  = self.calculate_Features(field_real_B)
        out7_f_A, out14_f_A, out23_f_A  = self.calculate_Features(field_fake_A)
        self.field_real_B = field_real_B

        self.loss_F_B_field, self.loss_E1_field, self.loss_E2_field, self.loss_E3_field= self.criterionFeatureField(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)


        out7_r_B, out14_r_B, out23_r_B  = self.calculate_Features(robot_line_goal_real_B)
        out7_f_A, out14_f_A, out23_f_A  = self.calculate_Features(robot_line_goal_fake_A)
        self.loss_F_B_RLG, self.loss_E1_RLG, self.loss_E2_RLG, self.loss_E3_RLG = self.criterionFeature(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)

        self.loss_F_B = self.loss_F_B_field + self.loss_F_B_RLG
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = 0#self.criterionCycle_B(self.rec_B, self.real_B) * lambda_B

        # GAN loss D_B(G_B(B))


        self.loss_G_B_G = self.criterionGAN(self.netD_B_G(self.fake_A), True)
        self.loss_G_B_L = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_G_B = self.loss_G_B_G * 0.2 + self.loss_G_B_L * 0.8
        
        #test = self.real_B_channels[0,2:5,:,:]



        

        #recBB = util.tensor2im(self.smooth_fake_A)
        #util.save_image(recBB, "self.smooth_fake_A.png")

        #--------------------------------Total------------------------------------------------------

        self.CycleLoss = self.loss_cycle_A + self.loss_cycle_B

        self.D_Loss = self.loss_G_A + self.loss_G_B

        # combined loss and calculate gradients
        self.loss_G = self.CycleLoss  + self.D_Loss + self.loss_F_A +self.loss_F_B + self.loss_idt_B + self.loss_idt_A
        self.loss_G.backward()

    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netF], False)
        self.set_requires_grad([self.netD_B, self.netD_B_G], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_B, self.netD_B_G], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_G.zero_grad()   # set D_A and D_B's gradients to zero
        #self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.optimizer_D_G.step()  # update D_A and D_B's weights
