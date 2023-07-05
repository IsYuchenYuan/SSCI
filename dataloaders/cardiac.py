
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict
import nibabel as nib
from torchvision import transforms
import cv2
from skimage import transform
import math
from scipy.ndimage import zoom
from scipy import ndimage


class Cardiac(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train_l', percentage=0.1, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        with open(self._base_dir + '/train_c.txt', 'r') as f:
            lists = f.readlines()
        trainlists = lists[:-4]
        random.seed(2)  # 1
        random.shuffle(trainlists)
        label_num = int(len(trainlists) * percentage)
        if split=='train_l':
            self.image_list = trainlists[:label_num]
        elif split=='train_ul':
            self.image_list = trainlists[label_num:]
        elif split == 'test':
            self.image_list = lists[-4:]
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/cardiac_h5py/"+image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def read_info(path):
    f = open(path, 'r')
    Lines = f.readlines()
    info = defaultdict(list)
    for line in Lines:
        res = dict(map(lambda x: x.split(': '), line.rstrip("\n").split(",")))
        info.update(res)
    return info

class ACDC(Dataset):

    def __init__(self, data_dir,split='train_l', percentage=1):
        """

        :param data_dir:
        :param split:
        :param percentage:
        :param phase: 'ED' or 'ES' or 'all'
        """
        super().__init__()
        self.pathsize=256
        image_path = []
        with open('../data/ACDC/' + 'train.txt', 'r') as f:
            lists = f.readlines()
        trainlists = lists[:80]

        # keep the random seed consistent for all the experiments
        random.seed(2)
        random.shuffle(trainlists)

        label_num = math.ceil(len(trainlists) * percentage)
        if split == 'train_l':
            self.image_list = trainlists[:label_num]
        elif split == 'train_ul':
            self.image_list = trainlists[label_num:]
        elif split == 'test':
            self.image_list = lists[80:]
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("total {} samples".format(len(self.image_list)))

        for patient in self.image_list:

            info = read_info(data_dir + patient + '/Info.cfg')
            ed = data_dir + patient + '/' + patient + '_frame' + str(
                info['ED']).zfill(2)
            img = nib.load(ed + '.nii.gz')
            for i in range(img.get_fdata().shape[2]):
                image_path.append([ed + '.nii.gz', ed + '_gt.nii.gz', i])

            es = data_dir + patient + '/' + patient + '_frame' + str(
                info['ES']).zfill(2)
            img = nib.load(es + '.nii.gz')
            for i in range(img.get_fdata().shape[2]):
                image_path.append([es + '.nii.gz', es + '_gt.nii.gz', i])

        random.shuffle(image_path)
        assert len(image_path) > 0
        print(len(image_path))
        self.image_path = image_path
        if split == "test":
            self.transform = transforms.Compose([ToTensor_2d()])
        else:
            self.transform = transforms.Compose([RandomGenerator(), ToTensor_2d()])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = nib.load(self.image_path[item][0])
        image = img.get_fdata() # 'numpy.ndarray'
        img_dat = img.header
        gt = nib.load(self.image_path[item][1]).get_fdata()

        image = (image - np.mean(image)) / np.std(image)

        image = image[:, :, self.image_path[item][2]].squeeze()
        gt = gt[:, :, self.image_path[item][2]].squeeze()

        ## preprocessing Method1 --- zoom
        # x,y = image.shape
        # image = zoom(image, (self.pathsize / x, self.pathsize / y), order=0)
        # gt = zoom(gt, (self.pathsize / x, self.pathsize / y), order=0)

        ###### preprocessimg Method2 -- resample

        pixel_size = (img_dat.structarr['pixdim'][1], img_dat.structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / 1,
                        pixel_size[1] / 1)
        image = transform.rescale(image,
                                  scale_vector,
                                  order=1,
                                  preserve_range=True,
                                  anti_aliasing=True,
                                  mode='constant')
        gt = transform.rescale(gt,
                               scale_vector,
                               order=0,
                               preserve_range=True,
                               mode='constant')

        image = crop_or_pad_slice_to_size(image, self.pathsize, self.pathsize, fillvalue=np.min(image))
        gt = crop_or_pad_slice_to_size(gt, self.pathsize, self.pathsize, fillvalue=np.min(gt))

        sample = {'image': image, 'label': gt}
        sample = self.transform(sample)
        return sample


class Normalization(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        imgs_std = np.std(image)
        imgs_mean = np.mean(image)
        imgs_normalized = (image - imgs_mean) / imgs_std
        return {'image': imgs_normalized, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class Resize(object):
    def __init__(self, size):
        assert type(size) in [int, tuple], "CHECK SIZE TYPE!"
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'label': label}

def crop_or_pad_slice_to_size(slice, nx, ny,fillvalue):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.full((nx, ny),fill_value=fillvalue)
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        sample = {'image': image, 'label': label}
        return sample


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomRot(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)

        return {'image': image, 'label': label}



class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class ToTensor_2d(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.astype(np.float32)
        image = image.reshape(1, image.shape[0], image.shape[1]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


# class SmoothLable(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         label = sample['label']
#         image = sample['image']
#         cls_id_map = [1, 2, 3, 4, 5, 6, 7]
#         smooth_label = smooth_GT_label(GT_label_original=label, cls_map=cls_id_map, K_size=[2,2,2])
#         return {'image': image, 'label': smooth_label}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """
        grouper(primary_iter, self.primary_batch_size): split primary_iter into chunks with the size of primary_batch_size
        e.g., grouper([1,2,3,4], 2) --> [(1,2),(3,4)]
        :return:
        """
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
