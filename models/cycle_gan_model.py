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

        self.loss_names = ['D_B', 'D_B_G', 'G_B', 'G_B_L','G_B_G', 'F_B', 'F_B_field', 'F_B_ImageLayer', 'F_B_ImageLine', 'F_B_shirt', 'background_penalization', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []
        visual_names_B = ['real_B', 'fake_A']
        if self.isTrain:
            visual_names_A = []

            visual_names_B = ['real_B', 'fake_A', 'norm_field_real_B', 'norm_field_fake_A', 'Image_real_B', 'Image_fake_A', 'shirt_real_B', 'shirt_fake_A', 'back_fake_A', 'greenBack',       'imageLine_real_B', 'imageLine_fake_A']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            print("------------idt is used-------------")
            #visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_B', 'D_B', 'D_B_M', 'D_B_G']
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


        self.ones = torch.ones([opt.crop_size,opt.crop_size]).cuda().float()
        self.ones3D = torch.ones([1, opt.crop_size,opt.crop_size]).cuda().float()
        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'cascade', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            #self.netD_A = networks.define_D(opt.output_nc, opt.ndf, "pixel",
                                            #opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            self.netD_B = networks.define_D(387, 72, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, False)

            self.netD_B_M = networks.define_D(387, 72, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, False)

            self.netD_B_G = networks.define_D(387, 72, "n_layers", 
                                            3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, False)

            self.netF = networks.define_F(self.gpu_ids)
 

        self.backgroundFactor = 0.15
        self.foregroundFactor = 0.15 
        self.bound_back = 0.0
        self.bound_back_div = -0.05
        self.lay0 = torch.nn.InstanceNorm2d(3, affine=True).cuda()
        self.lay1 = torch.nn.LayerNorm([3, opt.crop_size, opt.crop_size], elementwise_affine=False).cuda()
        self.lay2 = torch.nn.LayerNorm([6, opt.crop_size, opt.crop_size], elementwise_affine=False).cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            
            factorField = 12/1000000
            self.factorBack = 1
            factoShirt = 1/100
            factorImage = 4/10000
            factorLines = 4/10000
                
            

            self.criterionFeatureField = networks.FeatureLoss(5.2*factorField , 10.8*factorField, 0.8*factorField, "L1").to(self.device)
            self.criterionFeatureImage =      networks.FeatureLoss(5.2*factorImage ,      10.8*factorImage,      0.8*factorImage,      "MSE").to(self.device)
            self.criterionFeatureLines =      networks.FeatureLoss(5.2*factorLines ,      10.8*factorLines,      0.8*factorLines,      "MSE").to(self.device)

            self.criterionFeatureShirt =      networks.FeatureLoss(5.2*factoShirt ,      10.8*factoShirt,      0.8*factoShirt,      "L1").to(self.device)
            #self.criterionFeature = networks.FeatureLoss(1000000 , 1000000, 1000000, 1).to(self.device)

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle_B = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

            self.avg_pool_disc = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

            self.upsample = torch.nn.Upsample(size=opt.crop_size, mode='bilinear')
            self.upsample_G = torch.nn.Upsample(size=(int)(opt.crop_size/2), mode='bilinear')

            self.jitter = torchvision.transforms.ColorJitter(brightness=0.025, contrast=0.025, saturation=0.025, hue=0.015)
            self.jitterIdt = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters(),self.netD_B_M.parameters(),self.netD_B_G.parameters()), lr=opt.lr*2, betas=(opt.beta1, 0.999))



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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def infer(self, image):
        image = image* 2.0 / 255.0-1
        real_B_channels = self.add_background_field_channel(image, image)
        fake_A_array = self.netG_B.module(real_B_channels)# G_B(B)
        fake_A = ((fake_A_array[0,:]).unsqueeze(0).cpu().float() + 1)/ 2.0 * 255.0
        #fake_A = min(max(fake_A, 0.0), 255.0)
        return fake_A

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:

            
            self.real_B_channels = self.add_background_field_channel(self.real_B, self.real_B)
            self.fake_A_array = self.netG_B(self.real_B_channels)# G_B(B)

            self.fake_A = (self.fake_A_array[0,:]).unsqueeze(0)

        else:

            self.real_B_channels = self.add_background_field_channel(self.real_B, self.real_B)
            self.fake_A_array = self.netG_B(self.real_B_channels)# G_B(B)

            self.fake_A = (self.fake_A_array[0,:]).unsqueeze(0)

            self.set_requires_grad([self.netG_B], False)
            
            traced_script_module = torch.jit.trace(self.infer, self.real_B)
            traced_script_module.save("traced_unet_256.pt")
            


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
        
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Combined loss and calculate gradients
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
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

        out0_f_A, out1_f_A, out2_f_A, out3_f_A, _  = self.calculate_Features(fake_A, 'layer')
        out0_r_A, out1_r_A, out2_r_A, out3_r_A, _  = self.calculate_Features(real_A, 'layer')


        concat_fake_A = torch.cat([self.upsample(out3_f_A[:,:,:]), self.upsample(out2_f_A[:,:,:])], 1)
        concat_fake_A = torch.cat([fake_A, concat_fake_A], 1)

        concat_real_A = torch.cat([self.upsample(out3_r_A[:,:,:]), self.upsample(out2_r_A[:,:,:])], 1)
        concat_real_A = torch.cat([real_A, concat_real_A], 1)


        self.loss_D_B = self.backward_D_basic(self.netD_B, concat_real_A, concat_fake_A)

        self.loss_D_B_M = self.backward_D_basic(self.netD_B_M, self.avg_pool_disc(concat_real_A), self.avg_pool_disc(concat_fake_A))

        self.loss_D_B_G = self.backward_D_basic(self.netD_B_G, self.avg_pool_disc(self.avg_pool_disc(concat_real_A)), self.avg_pool_disc(self.avg_pool_disc(concat_fake_A)))

    def calculate_Features(self, image, normalize = 'none'):
        generated = (image+1.0)/2.0*255.0
        input_image = generated-self.bb
        if(normalize == 'instance'):
            input_image=self.lay0(input_image)
        elif(normalize == 'layer'):
            input_image=self.lay1(input_image)
 
        out0, out1, out2, out3, out4 =self.netF(input_image)
        return out0, out1, out2, out3, out4

    def backward_G(self,epoch):
        """Calculate the loss for generators G_A and G_B"""

        #--------------------------------cycle B------------------------------------------------------

#imageAux.unsqueeze(0), imageAuxTen.unsqueeze(0), field.unsqueeze(0), fieldTen.unsqueeze(0), background.unsqueeze(0), backGroundTen.unsqueeze(0), shirt.unsqueeze(0), shirtTen.unsqueeze(0)


        imageAux_real_B, imageAuxTen, field_real_B, fieldTen, back_real_B, backGroundTen, shirt_real_B, shirtTen, imageLine_real_B, lineTen = self.get_field_robot_line_goal_background_images(self.real_B,self.real_B)
        imageAux_fake_A, _          , field_fake_A, _       , back_fake_A, _            , shirt_fake_A,        _, imageLine_fake_A, _        = self.get_field_robot_line_goal_background_images(self.fake_A,self.real_B)
 


#------------------------------------------field---------------------------------

        self.norm_field_fake_A = self.lay1(field_fake_A)
        self.norm_field_real_B = self.lay1(field_real_B)

        _, out7_r_B, _, out14_r_B, out23_r_B  = self.calculate_Features(self.norm_field_real_B, 'layer')
        _, out7_f_A, _, out14_f_A, out23_f_A  = self.calculate_Features(self.norm_field_fake_A, 'layer')

        sumFieldTen = torch.sum(fieldTen).cpu().float().detach().numpy()
        self.loss_F_B_field = 0
        if(sumFieldTen!=0):
            self.loss_F_B_field, _, _, _= self.criterionFeatureField(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)/sumFieldTen

        



#--------------------------------back-----------------------------------------

        self.back_fake_A = back_fake_A
        self.back_real_B = back_real_B


#--------------------------------robot/goal-----------------------------------------
        

        self.Image_fake_A = imageAux_fake_A
        self.Image_real_B = imageAux_real_B
        _, out7_r_B, _, out14_r_B, out23_r_B  = self.calculate_Features(imageAux_real_B, 'layer')#poner le fondo de real B en fake A para la normalizacion
        _, out7_f_A, _, out14_f_A, out23_f_A  = self.calculate_Features(imageAux_fake_A, 'layer')



        
        sumimageAuxTen = torch.sum(imageAuxTen).cpu().float().detach().numpy()
        self.loss_F_B_ImageLayer = 0

        if(sumimageAuxTen!=0):
            self.loss_F_B_ImageLayer, _, _, _= self.criterionFeatureImage(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)/sumimageAuxTen
#--------------------------------lines-----------------------------------------
        

        self.imageLine_fake_A = self.lay1(imageLine_fake_A)
        self.imageLine_real_B = self.lay1(imageLine_real_B)
        _, out7_r_B, _, out14_r_B, out23_r_B  = self.calculate_Features(imageLine_real_B, 'layer')#poner le fondo de real B en fake A para la normalizacion
        _, out7_f_A, _, out14_f_A, out23_f_A  = self.calculate_Features(imageLine_fake_A, 'layer')



        
        sumimageLineTen = torch.sum(lineTen).cpu().float().detach().numpy()
        self.loss_F_B_ImageLine = 0

        if(sumimageLineTen!=0):
            self.loss_F_B_ImageLine, _, _, _= self.criterionFeatureLines(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)/sumimageLineTen


#--------------------------------shirt-----------------------------------------
        

        self.shirt_fake_A = self.lay1(shirt_fake_A)
        self.shirt_real_B = self.lay1(shirt_real_B)

        _, out7_r_B, _, out14_r_B, out23_r_B  = self.calculate_Features(shirt_real_B, 'layer')#poner le fondo de real B en fake A para la normalizacion
        _, out7_f_A, _, out14_f_A, out23_f_A  = self.calculate_Features(shirt_fake_A, 'layer')

        sumshirtTen = torch.sum(shirtTen).cpu().float().detach().numpy()
        self.loss_F_B_shirt = 0

        if(sumshirtTen!=0):
            self.loss_F_B_shirt, _, _, _= self.criterionFeatureShirt(out7_r_B, out14_r_B, out23_r_B, out7_f_A, out14_f_A, out23_f_A)/sumshirtTen

# ----------------------------------GAN loss D_B(G_B(B))-----------------------------------

        out0_f_A, _, out2_f_A, out3_f_A, _  = self.calculate_Features(self.fake_A, 'layer')


        concat_fake_A = torch.cat([self.upsample(out3_f_A[:,:,:]), self.upsample(out2_f_A[:,:,:])], 1)
        concat_fake_A = torch.cat([self.fake_A, concat_fake_A], 1)



        self.loss_G_B_L = self.criterionGAN(self.netD_B(concat_fake_A), True)
        self.loss_G_B_M = self.criterionGAN(self.netD_B_M(self.avg_pool_disc(concat_fake_A)), True)
        self.loss_G_B_G = self.criterionGAN(self.netD_B_G(self.avg_pool_disc(self.avg_pool_disc(concat_fake_A))), True)

        


        #const_aux = max(epoch,self.opt.n_epochs)-self.opt.n_epochs
        #const=const_aux/self.opt.n_epochs_decay*0.4
        self.loss_G_B = self.loss_G_B_G * 1/4 + self.loss_G_B_L * 1/4 + self.loss_G_B_M * 2/4
        #self.loss_G_B  = self.loss_G_B_L


#---------------------background--------------------------------

        self.loss_background_penalization = self.penalize_background(back_fake_A, backGroundTen)*8 #self.bound_back
            #self.loss_background_penalization += self.penalize_background(back_fake_A, 74, 54, 44, ba8ckGroundTen)


        

#--------------------------------Total------------------------------------------------------

        self.loss_F_B = self.loss_F_B_field + self.loss_F_B_ImageLayer + self.loss_F_B_shirt + self.loss_F_B_ImageLine
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_B + self.loss_F_B  + self.loss_background_penalization
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
        #self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
















    def penalize_background(self, image, backGroundTen):
        image = image.squeeze(0)
        image = self.avg_pool(image)
        back = backGroundTen.squeeze(0)
        #R = Rin * 2.0 / 255.0
        #R = R - 1
        G = 75 * 2.0 / 255.0
        G2 = 140 * 2.0 / 255.0
        #G = G - 1
        #B = Bin * 2.0 / 255.0
        #B = B - 1

        GR = 1.1#G/R
        GB = 1.3#G/B
        #print(GR, GB)
        channelG = image[1,:,:]+self.ones
        #R
        aaa = self.ones*GR
        channelR = image[0,:,:]+self.ones
        division = channelG/channelR - aaa
        
        booleanTenR = division > self.bound_back_div#0.02
     
        #G
        aaa = self.ones*G
        #substraction = torch.abs(channelG - aaa)
        booleanTenG = channelG > aaa + self.bound_back#0.02

        #G
        aaa = self.ones*G2
        #substraction = torch.abs(channelG - aaa)
        booleanTenG2 = channelG < aaa - self.bound_back#0.02
        

        #B
        aaa = self.ones*GB
        channelB = image[2,:,:]+self.ones
        division = channelG/channelB - aaa
        booleanTenB = division > self.bound_back_div#0.02

        boolBadG = booleanTenR & booleanTenB & backGroundTen & booleanTenG & booleanTenG2



        #R = Rin * 2.0 / 255.0
        #R = R - 1
        R = 80 * 2.0 / 255.0
        #G = G - 1
        #B = Bin * 2.0 / 255.0
        #B = B - 1

        RG = 1.5#G/R
        RB = 1.5#G/B
        #print(GR, GB)
        channelR = image[0,:,:]+self.ones
        #G
        aaa = self.ones*RG
        channelG = image[1,:,:]+self.ones
        division = channelR/channelG - aaa
        
        booleanTenG = division > self.bound_back_div#0.02
     
        #R
        aaa = self.ones*R
        #substraction = torch.abs(channelG - aaa)
        booleanTenR = channelR > aaa + self.bound_back#0.02

        #B
        aaa = self.ones*RB
        channelB = image[2,:,:]+self.ones
        division = channelR/channelB - aaa
        booleanTenB = division > self.bound_back_div#0.02

        boolBadR = booleanTenR & booleanTenB & backGroundTen & booleanTenG

        boolBad = boolBadG | boolBadR
        
        sumboolBad = torch.sum(boolBad).cpu().float().detach().numpy()
        
        sumboolBack = torch.sum(backGroundTen).cpu().float().detach().numpy()
        greenBack = image*boolBad
        self.greenBack = greenBack.unsqueeze(0)
        if(sumboolBack==0):
            return 0
        return sumboolBad/(sumboolBack)




    def add_background_field_channel(self, image, label):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 0 * 2.0 / 255.0
        R = R - 1
        G = 0 * 2.0 / 255.0
        G = G - 1
        B = 0 * 2.0 / 255.0
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


        R = 76 * 2.0 / 255.0
        R = R - 1
        G = 93 * 2.0 / 255.0
        G = G - 1
        B = 57 * 2.0 / 255.0
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

    def get_field_robot_line_goal_background_images(self, image, label):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 0 * 2.0 / 255.0
        R = R - 1
        G = 0 * 2.0 / 255.0
        G = G - 1
        B = 0 * 2.0 / 255.0
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
        background = image * backGroundTen
        #print("background",background.size())
        

        

#----------------------------------------field-----------------------------------------------


        R = 76 * 2.0 / 255.0
        R = R - 1
        G = 93 * 2.0 / 255.0
        G = G - 1
        B = 57 * 2.0 / 255.0
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
        notFieldTen = ~fieldTen

#------------------------------------------shirt--------------------------------------------------------

        R = 60 * 2.0 / 255.0
        #G = G - 1
        #B = Bin * 2.0 / 255.0
        #B = B - 1

        RG = 2#G/R
        RB = 2#G/B
        #print(GR, GB)
        channelR = label[0,:,:]+self.ones
        #G
        aaa = self.ones*RG
        channelG = label[1,:,:]+self.ones
        division = channelR/channelG - aaa
        
        booleanTenG = division > self.bound_back_div#0.02
     
        #R
        aaa = self.ones*R
        #substraction = torch.abs(channelG - aaa)
        booleanTenR = channelR > aaa + self.bound_back#0.02

        #B
        aaa = self.ones*RB
        channelB = label[2,:,:]+self.ones
        division = channelR/channelB - aaa
        booleanTenB = division > self.bound_back_div#0.02

        shirtTen = booleanTenR & booleanTenB & booleanTenG
        notShirt = ~shirtTen
        shirt = shirtTen*image

        #----------------------------------------lines-----------------------------------------------


        R = 224 * 2.0 / 255.0
        R = R - 1
        G = 224 * 2.0 / 255.0
        G = G - 1
        B = 215 * 2.0 / 255.0
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
        

        lineTen = booleanTenR * booleanTenG * booleanTenB
        notLineTen = ~fieldTen

#------------------------------------------shirt--------------------------------------------------------



        #robots_goal_lines = image * notFieldTen * notbackGroundTen * notShirt
        #robots_goal_lines = robots_goal_lines - self.ones * fieldTen - self.ones * backGroundTen
        #print("foreground",foreground.size())
        field = image * fieldTen
        field = field - self.ones * notFieldTen

        imageAuxTen = notbackGroundTen * notFieldTen * notShirt

        imageAux = image * imageAuxTen + label * fieldTen - self.ones * backGroundTen

        imageLine = image * lineTen
        return imageAux.unsqueeze(0), imageAuxTen.unsqueeze(0), field.unsqueeze(0), fieldTen.unsqueeze(0), background.unsqueeze(0), backGroundTen.unsqueeze(0), shirt.unsqueeze(0), shirtTen.unsqueeze(0), imageLine.unsqueeze(0), lineTen.unsqueeze(0)




    def remove_background(self, image, label, bound):
        label = label.squeeze(0)
        image = image.squeeze(0)
        R = 0 * 2.0 / 255.0
        R = R - 1
        G = 0 * 2.0 / 255.0
        G = G - 1
        B = 0 * 2.0 / 255.0
        B = B - 1

        #R
        aaa = self.ones*R
        channelR = label[0,:,:]
        substraction = torch.abs(channelR - aaa)
        booleanTenR = substraction < bound#0.02
     
        #G
        aaa = self.ones*G
        channelG = label[1,:,:]
        substraction = channelG - aaa
        booleanTenG = channelG > aaa - bound#0.02

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
        R = 0 * 2.0 / 255.0
        R = R - 1
        G = 0 * 2.0 / 255.0
        G = G - 1
        B = 0 * 2.0 / 255.0
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
        #print("sum boolBack------------------", torch.sum(boolBack).cpu().float().detach().numpy())
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
