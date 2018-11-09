import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from generate_PSF import PSF
from generate_trajectory import Trajectory

import sys
import time
from joblib import Parallel, delayed
import multiprocessing

import pdb

class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """
        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = misc.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images yet.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(
                    canvas=self.shape[0], 
                    path_to_save=os.path.join(
                        self.path_to_save,
                        'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []


    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        
        result=[]
        if len(psf) > 1:
            for p in psf:

                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                    
                for i in range(3):
                    blured[:, :, i] = cv2.filter2D(blured[:, :, i], -1, tmp)

                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))

        else:

            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)

            for i in range(3):
                blured[:, :, i] = cv2.filter2D(blured[:, :, i], -1, tmp)
            
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.abs(blured))

        self.result = result

        if self.path_to_save is None:
            raise Exception('Please create trajectory instance with path_to_save')

        cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)


def blur(args):

    # Unpack args
    img_path, folder_in, folder_out = args 
    print(img_path)

    # Blur images
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()

    psf = PSF(canvas=64, trajectory=trajectory).fit()

    BlurImage(
        os.path.join(folder_in, img_path), 
        PSFs=psf, 
        path__to_save=folder_out, 
        part=np.random.choice([1, 2, 3])).\
            blur_image(save=True)


if __name__ == '__main__':

    folder_in = sys.argv[1]
    if not folder_in[-1] == '/': 
        folder_in = folder_in + '/'

    if not os.path.exists(folder_in):
        raise Exception('Dude, that\'s not a real folder')

    folder_out = folder_in[0:-1] + '_blurred/'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    print ("\nINPUT:",folder_in)
    print ("OUTPUT:",folder_out,"\n") 
    
    # Run in parallel
    if len(sys.argv) > 2:

        desired_cores = int(sys.argv[2])
        assert desired_cores > 0
        
        num_cores = min(multiprocessing.cpu_count(), desired_cores)
        print ("Using {} cores...".format(num_cores))

        Parallel(n_jobs=num_cores)(delayed(blur)(
            [img_path, folder_in, folder_out]) for img_path in os.listdir(folder_in))

    # Run in Serial
    else:

        for img_path in os.listdir(folder_in):
            blur([img_path, folder_in, folder_out])
