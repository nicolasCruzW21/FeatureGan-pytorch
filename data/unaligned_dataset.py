import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.label=False
        self.domain=True
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A/original')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B/original')  # create a path '/path/to/data/trainB'
        self.dir_B_models = os.path.join(opt.dataroot, opt.phase + 'B/models')  # create a path '/path/to/data/trainB'






        #self.dir_D = os.path.join(opt.dataroot, 'domain')  # create a path '/path/to/data/trainB'
 
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.B_models_paths = sorted(make_dataset(self.dir_B_models, opt.max_dataset_size))    # load images from '/path/to/data/trainB'




        #self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.B_models_size = len(self.B_models_paths)  # get the size of dataset B

        B_model = Image.open(self.B_models_paths[0]).convert('RGB')
        transform_params = get_params(self.opt, B_model.size)
        self.transform_Model = get_transform(self.opt, transform_params, normalize =True, grayscale=False, rotate = True)
        self.transform_B_model = self.transform_Model(B_model).unsqueeze(0)
        #self.transform_B_models = self.transform_B_model
        print("self.B_models_size", self.B_models_size)
        #for i in range(1, self.B_models_size):
            #B_model = Image.open(self.B_models_paths[i]).convert('RGB')
            #self.transform_B_model = self.transform_Model(B_model).unsqueeze(0)
            #self.transform_B_models = torch.cat([self.transform_B_models, self.transform_B_model],0)
            
        #print(self.transform_B_models.shape)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.original_crop = opt.crop_size
        if(not self.label):
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), rotate = True)
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), rotate = True)
            
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        B_path_blured = B_path.replace("original","blured")
        B_path_depth = B_path.replace("original","depth")
        B_path_mask = B_path.replace("original","mask")


        A_path_blured = A_path.replace("original","blured")
        A_path_depth = A_path.replace("original","depth")
        A_path_mask = A_path.replace("original","mask")

        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #A_blured = Image.open(A_path_blured).convert('RGB')
        A_depth = Image.open(A_path_depth).convert('L')
        A_mask = Image.open(A_path_mask).convert('L')

        #B_blured = Image.open(B_path_blured).convert('RGB')
        B_depth = Image.open(B_path_depth).convert('L')
        B_mask = Image.open(B_path_mask).convert('L')


        # apply image transformation
        rotate = True
        if (rotate):
            rotate = random.randint(-6, 6)
            A_img = TF.rotate(A_img, rotate)
            B_img = TF.rotate(B_img, rotate)

            #A_blured_img = TF.rotate(A_blured, rotate)
            #B_blured_img = TF.rotate(B_blured, rotate)

            A_depth_img = TF.rotate(A_depth, rotate)
            B_depth_img = TF.rotate(B_depth, rotate)

            A_mask_img = TF.rotate(A_mask, rotate)
            B_mask_img = TF.rotate(B_mask, rotate)
        
        transform_params = get_params(self.opt, B_img.size)


        self.transform_B = get_transform(self.opt, transform_params, normalize =True, grayscale=False, rotate = True)
        self.transform_A = get_transform(self.opt, transform_params, normalize =True, grayscale=False, rotate = True)

        self.transform_L = get_transform(self.opt, transform_params, normalize =True, grayscale=True, rotate = True, method=Image.NEAREST)
        self.transform_M = get_transform(self.opt, transform_params, normalize =False, grayscale=True, rotate = True)


        #print("---------------------------------------------")
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        A_depth = self.transform_L(A_depth_img)
        B_depth = self.transform_L(B_depth_img)

        #A_blured = self.transform_A(A_blured_img)
        #B_blured = self.transform_B(B_blured_img)

        A_mask = self.transform_M(A_mask_img)
        B_mask = self.transform_M(B_mask_img)


        self.opt.crop_size = self.original_crop
        return {'A': A, 'B': B, \
        'A_depth': A_depth, 'B_depth': B_depth, \
        'A_mask': A_mask, 'B_mask': B_mask,\
        'A_paths': A_path, 'B_paths': B_path}#,\
        #'A_models': self.transform_B_models, 'B_models': self.transform_B_models}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
