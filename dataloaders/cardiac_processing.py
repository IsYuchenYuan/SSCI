import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib
import os
output_size =[112, 112, 80]


def covert_h5():
    listt = glob('../../data/MMWHS/mr_train/*_image.nii.gz')
    for item in tqdm(listt):
        image = nib.load(item).get_data()
        image = np.array(image)
        print(image.shape)

        label = nib.load(item.replace('_image.nii.gz', '_label.nii.gz')).get_data()
        label = np.array(label)
        """
        myocardium of the left ventricle (MYO):205 --> 1
        left atrium blood cavity (LA): 420 --> 2
        left ventricle blood cavity (LV): 500 --> 3
        right atrium blood cavity (RA): 550 --> 4
        right ventricle blood cavity (RV): 600 --> 5
        ascending aorta (AA): 820 --> 6
        pulmonary artery (PA): 850 --> 7
        """
        label[label == 205] = 1
        label[label == 420] = 2
        label[label == 500] = 3
        label[label == 550] = 4
        label[label == 600] = 5
        label[label == 820] = 6
        label[label == 850] = 7
        label = label.astype(np.uint8)
        w, h, d = label.shape
        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        num = np.percentile(image, 98)
        image = np.clip(image,0,num)
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        path = '../../data/cardiac_h5py/'
        if not os.path.exists(path):
            os.makedirs(path)
        h5path = os.path.join(path, item.split("/")[-1].replace('_image.nii.gz', '.h5'))
        print(h5path)
        f = h5py.File(h5path, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()




if __name__ == '__main__':
    covert_h5()
