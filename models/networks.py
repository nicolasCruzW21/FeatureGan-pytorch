import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import os, scipy.io
import numpy as np
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_12blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12)
    elif netG == 'resnet_16blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=16)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnetPyramid_9blocks':
        net = ResnetPyramidGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnetPyramid_6blocks':
        net = ResnetPyramidGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnetPyramid_16blocks':
        net = ResnetPyramidGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=16)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'cascade':
        net = cascaded_model(input_nc, output_nc , 256, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def vggnet(pretrained=False, model_root=None, **kwargs):
    model = VGG19(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model

def define_F(gpu_ids=[]):

    Net=vggnet(pretrained=False, model_root=None)
    Net=Net.eval()
    for param in Net.parameters():
        param.requires_grad = False

    Net=Net.to(gpu_ids[0])

    vgg_rawnet=scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')

    vgg_layers=vgg_rawnet['layers'][0]

    #Weight initialization according to the pretrained VGG Very deep 19 network Network weights

    layers=[0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]

    att=['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13', 'conv14', 'conv15', 'conv16']

    S=[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    for L in range(16):
        getattr(Net, att[L]).weight=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][0]).permute(3,2,0,1).cuda())
        getattr(Net, att[L]).bias=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][1]).view(S[L]).cuda())
    return Net.eval()







def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], alternate = False, with_statistics=True, pyramid = False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, alternate=alternate, with_statistics = with_statistics, pyramid = pyramid)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, alternate=alternate, with_statistics = with_statistics, pyramid = pyramid)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'pyramid':
        net = NLayerPyramidDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class FeatureLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, coef1, coef2, coef3, loss_type):
        """ Initialize the FeatureLoss class.
        
        """
        self.coef1 = coef1
        self.coef2 = coef2
        self.coef3 = coef3
        self.loss_type = loss_type
        super(FeatureLoss, self).__init__()
    def compute_error(self, R, F):
 
        if(self.loss_type == "MSE"):
            self.loss = torch.nn.MSELoss()
        elif(self.loss_type == "L1"):
            self.loss = torch.nn.L1Loss()
        else:
            self.loss = torch.nn.SmoothL1Loss()
        E = self.loss(R,F)
        return E

    def __call__(self, out7_r, out14_r, out23_r, out7_f, out14_f, out23_f):
        self.lay0 = torch.nn.InstanceNorm2d(64, affine=False).cuda()
        self.lay1 = torch.nn.InstanceNorm2d(64, affine=False).cuda()
        self.lay2 = torch.nn.InstanceNorm2d(128, affine=False).cuda()


        E1=self.compute_error(self.lay0(out7_r.unsqueeze(0)), self.lay0(out7_f.unsqueeze(0)))
        E2=self.compute_error(self.lay1(out14_r.unsqueeze(0)), self.lay1(out14_f.unsqueeze(0)))
        E3=self.compute_error(self.lay2(out23_r.unsqueeze(0)), self.lay2(out23_f.unsqueeze(0)))
        
        Total_loss= max(E1/self.coef1 + E2/self.coef2 + E3/self.coef3,0)
        return Total_loss, E1/self.coef1, E2/self.coef2, E3/self.coef3



class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=0.9, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class ResnetCascadeGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=68, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetCascadeGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        feature_size=input_nc

        model_encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]

        self.model_encoder = nn.Sequential(*model_encoder)


        mult = 2 ** n_downsampling
        model_resnet=[norm_layer(ngf * mult + feature_size)]
        for i in range(n_blocks):       # add ResNet blocks

            model_resnet += [ResnetBlock(ngf * mult + feature_size, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model_resnet = nn.Sequential(*model_resnet)




        mult = 2 ** (n_downsampling)
        model_decoder = [nn.ConvTranspose2d(ngf * mult+feature_size, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        for i in range(1,n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model_decoder += [nn.ReflectionPad2d(3)]
        model_decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #model_decoder += [nn.Tanh()]
        self.tanh=nn.Tanh()

        self.model_decoder = nn.Sequential(*model_decoder)

    def forward(self, input):
        """Standard forward"""
        #features=self.upsample(input[:,3:,:,:])
        encoded = self.model_encoder(input)
        #print(encoded.size())
        upsample = torch.nn.Upsample(size=encoded.size()[2], mode='nearest').cuda()
        encoded_concat= torch.cat([encoded, upsample(input[:,:,:,:])],1)
        output = self.model_decoder(self.model_resnet(encoded_concat))
  

        return torch.cat([self.tanh(output[:,:3,:,:]),output[:,3:,:,:]],1)

class ResnetPyramidGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetPyramidGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input = torch.zeros([1,ngf-input_nc, 256, 256]).float().cuda()

        model_preprocess = [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model_preprocess += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model_direct = []
        for i in range(4):       # add ResNet blocks
            model_direct += [ResnetBlock((int)(ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


       
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model_pyramid_0 = [PyramidBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_pyramid_0 = nn.Sequential(*model_pyramid_0)
        mult = 2 ** 1
        model_pyramid_1 = [PyramidBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_pyramid_1 = nn.Sequential(*model_pyramid_1)
        mult = 2 ** 2
        model_pyramid_2 = [PyramidBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_pyramid_2 = nn.Sequential(*model_pyramid_2)




        mult = 2 ** (n_downsampling - 0)
        model_pyramid_2_up = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
        self.model_pyramid_2_up = nn.Sequential(*model_pyramid_2_up)

        mult = 2 ** (n_downsampling - 1)
        model_pyramid_1_up = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
        self.model_pyramid_1_up = nn.Sequential(*model_pyramid_1_up)

        mult = 2 ** (n_downsampling - 2)
        model_pyramid_0_up = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
        self.model_pyramid_0_up = nn.Sequential(*model_pyramid_0_up)

        #end_model = []
        #for i in range(2):       # add ResNet blocks
            #end_model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        #end_model += [nn.ReflectionPad2d(0)]
        #end_model += [norm_layer(ngf),nn.ReLU(True)]
        #end_model += [nn.Conv2d(ngf, ngf, kernel_size=1, padding=0)]
        out_model = []
        for i in range(4):       # add ResNet blocks
            out_model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.tanh = nn.Tanh()
        self.model_preprocess = nn.Sequential(*model_preprocess)
        self.model_direct = nn.Sequential(*model_direct)
        self.end_model = nn.Sequential(*end_model)
        self.out_model = nn.Sequential(*out_model)

    def forward(self, input):
        """Standard forward"""

        input2 = torch.cat([self.input,input],1)

        preprocess = self.model_preprocess(input2)


        direct = self.model_direct(preprocess)
        pyramid_0 = self.model_pyramid_0(preprocess)
        pyramid_1 = self.model_pyramid_1(pyramid_0)
        pyramid_2 = self.model_pyramid_2(pyramid_1)

        pyramid_2_up = self.model_pyramid_2_up(pyramid_2) + pyramid_1
        pyramid_1_up = self.model_pyramid_1_up(pyramid_2_up) + pyramid_0



        output = self.model_pyramid_0_up(pyramid_1_up) + direct


        return torch.cat([self.tanh(self.out_model(output)[:,:3,:,:]), output[:,3:32,:,:]],1)

class PyramidBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(PyramidBlock, self).__init__()
        self.pyramid_block = self.build_pyramid_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_pyramid_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        
        pyramid_block = [nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(dim * 2),
                  nn.LeakyReLU(True)]
        pyramid_block += [ResnetBlock(dim * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        pyramid_block += [ResnetBlock(dim * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        pyramid_block += [ResnetBlock(dim * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        pyramid_block += [ResnetBlock(dim * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        return nn.Sequential(*pyramid_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.pyramid_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(True)]
            model += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ngf, kernel_size=1, padding=0, bias=use_bias)]
        #print(ngf,"----------------------")
        end_model = [norm_layer(int(ngf)), nn.ReLU(True)]
        end_model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        end_model += [nn.Conv2d(ngf, 3, kernel_size=3, padding=1)]
        end_model += [nn.Tanh()]
        #model += [nn.Tanh()]
        #end_model=nn.Tanh()

        self.model = nn.Sequential(*model)
        self.end_model = nn.Sequential(*end_model)

    def forward(self, input):
        """Standard forward"""



        output=self.model(input)
        #print(output.size(),"-------------------------------")

        return torch.cat([self.end_model(output),output[:,3:29,:,:]],1)




class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.tanh = nn.Tanh()
    def forward(self, input):
        """Standard forward"""
        out = self.model(input)
        return torch.cat([self.tanh(out[:,:3,:,:]), out[:,3:,:,:]],1)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, alternate=0, with_statistics=True, pyramid = False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d



        kw = 4
        padw = 1
        sequence = [nn.Conv2d(ndf+3, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]#input_nc = 9
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            for l in range(0, alternate):
                print("----------------------------alternate-----------------------------",l+1)
                sequence += [
                    nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        squeeze=[]
        if(not with_statistics):
            squeeze = [torch.nn.InstanceNorm2d(input_nc-3)]
            print("without stats")
        squeeze+=[nn.Conv2d(input_nc-3, ndf, kernel_size=3, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        #squeeze += [nn.Conv2d(input_nc, 128, kernel_size=5, stride=1, padding=2)]
        #squeeze+= [nn.LeakyReLU(0.2, True)]
        #squeeze+=[torch.nn.InstanceNorm2d(128).cuda()]
        self.squeeze = nn.Sequential(*squeeze)

    def forward(self, input):
        """Standard forward."""
        
        input_no_RGB = input[:,3:,:]
        #print("input_no_RGB",input_no_RGB.size())
        squeezed = self.squeeze(input_no_RGB)
        #print("squeezed",squeezed.size())
        newInput = torch.cat([input[:,:3,:], squeezed], 1)


        return self.model(newInput), squeezed


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.lay0 = torch.nn.InstanceNorm2d(3, affine=True)
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1=nn.ReLU(inplace=True)
            
        self.conv2=nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2=nn.ReLU(inplace=True)
        self.max1=nn.AvgPool2d(kernel_size=3, stride=2)
        #128

            
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.lay3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.relu3=nn.ReLU(inplace=True)
            
        self.conv4=nn.Conv2d(128, 128,  kernel_size=3, padding=1, bias=True)
        self.relu4=nn.ReLU(inplace=True)
        self.max2=nn.AvgPool2d(kernel_size=3, stride=2)
#64



            
        self.conv5=nn.Conv2d(128, 256,  kernel_size=3, padding=1, bias=True)
        self.lay5 = torch.nn.InstanceNorm2d(256, affine=True)
        self.relu5=nn.ReLU(inplace=True)
            
        self.conv6=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.lay6 = torch.nn.InstanceNorm2d(256, affine=True)
        self.relu6=nn.ReLU(inplace=True)
            
        self.conv7=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.lay7 = torch.nn.InstanceNorm2d(256, affine=True)
        self.relu7=nn.ReLU(inplace=True)
            
        self.conv8=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.lay8 = torch.nn.InstanceNorm2d(256, affine=True)
        self.relu8=nn.ReLU(inplace=True)
        self.max3=nn.AvgPool2d(kernel_size=3, stride=2)
#32



            
        self.conv9=nn.Conv2d(256, 512,  kernel_size=3, padding=1, bias=True)
        self.lay9 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu9=nn.ReLU(inplace=True)
            
        self.conv10=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay10 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu10=nn.ReLU(inplace=True)
            
        self.conv11=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay11 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu11=nn.ReLU(inplace=True)
            
        self.conv12=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay12 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu12=nn.ReLU(inplace=True)
        self.max4=nn.AvgPool2d(kernel_size=3, stride=2)#16



            
        self.conv13=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay13 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu13=nn.ReLU(inplace=True)
            
        self.conv14=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay14 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu14=nn.ReLU(inplace=True)
            
        self.conv15=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay15 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu15=nn.ReLU(inplace=True)
            
        self.conv16=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.lay16 = torch.nn.InstanceNorm2d(512, affine=True)
        self.relu16=nn.ReLU(inplace=True)
        self.max5=nn.AvgPool2d(kernel_size=3, stride=2)#8

    def forward(self, x):
        
        out1= self.conv1(x)
        out2= self.relu1(out1)
            
        out3= self.conv2(out2)
        out4=self.relu2(out3)
        out5=self.max1(out4)


        out6=self.conv3(out5)
        out7=self.relu3(out6)     
        out8=self.conv4(out7)
        out9=self.relu4(out8)
        out10=self.max2(out9) 

         
        out11=self.conv5(out10)
        out12=self.relu5(out11)  
        out13=self.conv6(out12)
        out14=self.relu6(out13)           
        out15=self.conv7(out14)
        out16=self.relu7(out15)
        out17=self.conv8(out16)
        out18=self.relu8(out17)
        #out19=self.max3(out18)    
       
        #out20=self.conv9(out19)
        #out21=self.relu9(out20)
            
        #out22=self.conv10(out21)
        #out23=self.relu10(out22)
            
        #out24=self.conv11(out23)
        #out25=self.relu11(out24)         
        #out26=self.conv12(out25)
        #out27=self.relu12(out26)

        #out28=self.max4(out27)           
        #out29=self.conv13(out28)
        #out30=self.relu13(out29)
          
        #out31=self.conv14(out30)
        #out32=self.relu14(out31)
            
        #out33=self.conv15(out32)
        #out34=self.relu15(out33)
            
        #out35=self.conv16(out34)
        #out36=self.relu16(out35)

        #out37=self.max5(out36)

        return  out4, out7, out9, out18#, out27#, out36                   #Add appropriate outputs

