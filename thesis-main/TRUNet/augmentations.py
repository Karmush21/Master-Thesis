import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom #TODO Before. interpolation
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import time

from torchvision.transforms.functional import gaussian_blur


#aug_random = random.Random()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_augmentations(image, label, augmentations, prob=0.7):
    if random.random() > prob:
        return image, label  # No augmentation
    
    # Determine how many augmentations to apply (e.g., 1 to the number of augmentations available)
    num_augmentations = random.randint(1, len(augmentations))

    #print("here: " , num_augmentations)
    
    # Randomly select a subset of augmentations to apply
    selected_augmentations = random.sample(augmentations, num_augmentations)
    
    # Apply the selected augmentations
    for aug in selected_augmentations:
        image, label = aug(image, label)
    
    return image, label

#--------------------------


def random_rot_flip3d_torch(image, label):
    # Random rotations
    k1 = random.randint(0, 3)
    image = torch.rot90(image, k1, dims=(1, 0))
    label = torch.rot90(label, k1, dims=(1, 0))
    k2 = random.randint(0, 3)
    image = torch.rot90(image, k2, dims=(1, 2))
    label = torch.rot90(label, k2, dims=(1, 2))
    k3 = random.randint(0, 3)
    image = torch.rot90(image, k3, dims=(2, 0))
    label = torch.rot90(label, k3, dims=(2, 0))

    # Random flipping
    axis = random.randint(0, 2)
    image = torch.flip(image, dims=(axis,))
    label = torch.flip(label, dims=(axis,))

    return image, label

def random_rotate3d_torch(image, label):
    angles = [random.randint(-20, 20) for _ in range(3)]

    # Rotate around (height, width) axes
    image = F.rotate(image, angles[0], interpolation=F.InterpolationMode.BILINEAR)
    label = F.rotate(label, angles[0], interpolation=F.InterpolationMode.BILINEAR)

    # Rotate around (depth, width) axes
    image = F.rotate(image, angles[1], interpolation=F.InterpolationMode.BILINEAR)
    label = F.rotate(label, angles[1], interpolation=F.InterpolationMode.BILINEAR)

    # Rotate around (depth, height) axes
    image = F.rotate(image, angles[2], interpolation=F.InterpolationMode.BILINEAR)
    label = F.rotate(label, angles[2], interpolation=F.InterpolationMode.BILINEAR)

    return image, label

def gaussian_blur3d_torch(image, sigma_range=(1, 1.5)):
    sigma = random.uniform(*sigma_range)
    #print("SIGMA: ", sigma)
    # Assuming image is a tensor of shape (depth, height, width)
    image = gaussian_blur(image, kernel_size=3, sigma=sigma)

    return image


def aug_gaussian_blur_torch(image, label):
    return gaussian_blur3d_torch(image), label

def aug_random_rot_flip_torch(image, label):
    image, label = random_rot_flip3d_torch(image, label)
    return image, label

def aug_random_rotate_torch(image, label):
    image, label = random_rotate3d_torch(image, label)
    return image, label


# TODO I think you can remove these later. torch ones seem to work fine
# Found emperically that this range seems to work fine.
def gaussian_blur3d(image, sigma_range=(1, 1.5)):
    sigma = np.random.uniform(*sigma_range)
    image = gaussian_filter(image, sigma=sigma)
    return image


def random_rot_flip3d(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 0))
    label = np.rot90(label, k, axes=(1, 0))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(1, 2))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(2, 0))
    label = np.rot90(label, k, axes=(2, 0))
    axis = np.random.randint(0, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


# TODO Change the order to 3 for image. Better right?
# TODO Investigate order 3 on mask as well. 
def random_rotate3d(image, label):
    angle = np.random.randint(-20, 20)
    
    image = ndimage.rotate(image, angle, axes=(1, 0), order=3, reshape=False)
    label = ndimage.rotate(label, angle, axes=(1, 0), order=3, reshape=False)
    
    angle = np.random.randint(-20, 20)
    
    image = ndimage.rotate(image, angle, axes=(1, 2), order=3, reshape=False)
    label = ndimage.rotate(label, angle, axes=(1, 2), order=3, reshape=False)
    
    angle = np.random.randint(-20, 20)
    
    image = ndimage.rotate(image, angle, axes=(2, 0), order=3, reshape=False)
    label = ndimage.rotate(label, angle, axes=(2, 0), order=3, reshape=False)
    return image, label

def aug_gaussian_blur(image, label):
    return gaussian_blur3d(image), label

def aug_random_rot_flip(image, label):
    image, label = random_rot_flip3d(image, label)
    return image, label

def aug_random_rotate(image, label):
    image, label = random_rotate3d(image, label)
    return image, label



#Anmar: __call__: When the instance of a class is used as a function 

class RandomGenerator_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1]]

        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    
    #Anmar: Notice no self in the begining of these methods. Method is not bound to an instance of the class, but rather belongs to the class it self. 
    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return None, None

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class RandomGenerator3d_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, z = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        maxz = z - self.output_size[2]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        zstart = random.randint(0, maxz)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]

        if random.random() > 0.5:
            image, label = random_rot_flip3d(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate3d(image, label)

        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator3d_zoom(object):
    def __init__(self, output_size, use_aug):
        self.output_size = output_size
        self.use_aug = use_aug

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = torch.from_numpy(image.astype(np.float32)).to(device)
        label = torch.from_numpy(label.astype(np.float32)).to(device)

        x, y, z = image.shape
        # We already downsampled the volumes.
        if x != self.output_size[0] or y != self.output_size[1] or z != self.output_size[2]:
            print("Here???")
            quit()
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=0)

        if self.use_aug:
            augmentations = [
                aug_gaussian_blur_torch,
                aug_random_rot_flip_torch,
                aug_random_rotate_torch,
            ]
            image, label = apply_augmentations(image, label, augmentations, prob=0.70)

        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        image = image.cpu()
        label = label.cpu()

        sample = {'image': image, 'label': label}
        return sample



class Reshape3d_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        maxz = z - self.output_size[2]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        zstart = random.randint(0, maxz)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Reshape3d_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z),
                         order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        
        return sample


class Reshape_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomRotation3D(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label'] #TODO Change these later
        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # Randomly select rotation angles along x, y, and z axes
        angle_x = random.uniform(-self.degrees, self.degrees)
        angle_y = random.uniform(-self.degrees, self.degrees)
        angle_z = random.uniform(-self.degrees, self.degrees)

        # Apply rotations to both image and label
        # image_rotated = self.rotate3d(image, angle_x, angle_y, angle_z)
        # label_rotated = self.rotate3d(label, angle_x, angle_y, angle_z)

        image_rotated, label_rotated = random_rotate(image, label)
        
        #TODO These are numpy arrays now. perhaps we want them as torch tensors.
        
        # Convert tensors back to numpy arrays
        #image_rotated = image_rotated.numpy()
        #label_rotated = label_rotated.numpy()

        return {'image': image_rotated, 'label': label_rotated}

    def rotate3d(self, tensor, angle_x, angle_y, angle_z):
        # Convert tensor to numpy array
        tensor = tensor.numpy()

        # Rotate tensor along x-axis
        tensor = F.rotate(tensor, angle_x, mode='nearest', dims=(1, 2))
        # Rotate tensor along y-axis
        tensor = F.rotate(tensor, angle_y, mode='nearest', dims=(0, 2))
        # Rotate tensor along z-axis
        tensor = F.rotate(tensor, angle_z, mode='nearest', dims=(0, 1))

        return tensor
    


if __name__ == "__main__":
    torch.manual_seed(42)
    label = torch.rand(256,256, 100)
    image = torch.rand(256,256, 100)
    image, label = random_rotate3d(image, label)

    image = torch.from_numpy(image.astype(np.float32))
    label = torch.from_numpy(label.astype(np.float32))

    print(image.shape, label.shape)
    
    image = np.transpose(image, (2,0,1))
    label = np.transpose(label, (2,0,1))
    #image = image.permute(2, 0, 1)
    #label = label.permute(2, 0, 1)
    
    print(image.shape, label.shape)
    print(type(image), type(label))