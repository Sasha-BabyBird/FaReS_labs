import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
import os
import re

default_hist_param = 30
default_dft_param = 4
default_dct_param = 4
default_grad_param = 5
default_scale_param = 35

histogram = 'get_histogram'
dft = 'get_dft'
dct = 'get_dct'
gradient = 'get_gradient'
scale = 'get_scale'

class Image():
    def __init__(self, image_path, test=False):
        self.image_path = image_path
        self._label_regex = re.compile(r'(?<=s)\d+')
        #self._compiled_label_number = int(re.findall(self._label_regex, self.image_path)[0])
        self.label_number = int(re.findall(self._label_regex, self.image_path)[0])
        '''
        if test: 
            self.is_tested = True      
        else:
            self.is_tested = False
        '''
            


        number_regex = re.compile(r'\d+(?=\.pgm)')
        self.order_number = re.findall(number_regex, image_path)[0]
        self.matrix = cv2.imread(image_path, 0)
        
    '''
    def __setattr__(self, arg, new_value):
        if arg == 'is_tested' and new_value == False:
            if self.label_number is None or self.label_number == 0:
                self.label_number = self._compiled_label_number
                self.__dict__['is_tested'] = False
        elif arg == 'is_tested' and new_value == True:
            if self.label_number is not None or self.label_number == 0:
                self.label_number = None
                self.__dict__['is_tested'] = True
        else:
            self.__dict__[arg] = new_value
    '''
    

    def get_histogram(self, cols_number=default_hist_param):
        hist, _ = np.histogram(self.matrix, bins=np.linspace(0, 255, cols_number))
        return hist


    def get_dft(self, mat_size=default_dft_param):
        f = np.fft.fft2(self.matrix)
        f = f[0:mat_size, 0:mat_size]
        f = cv2.normalize(np.abs(f), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return f


    def get_dct(self, mat_size=default_dct_param):
        c = fftpack.dct(self.matrix, axis=1)
        c = fftpack.dct(c, axis=0)
        c = c[0:mat_size, 0:mat_size]
        c = cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return c


    def get_gradient(self, window_height=default_grad_param):
        rows_number = self.matrix.shape[0]
        i = 0
        features_number = rows_number - 2*window_height
        result = np.zeros(features_number)
        while i < features_number:
            upper_window = self.matrix[i:i+window_height, :]
            lower_window = self.matrix[i+window_height:i+2*window_height]
            result[i] = cv2.norm(upper_window, lower_window)
            i += 1
        return result


    def get_scale(self, scale=default_scale_param):
        h, w = self.matrix.shape
        new_size = (int(w * (scale / 100)), int(h * (scale / 100)))
        scaled_matrix = cv2.resize(self.matrix, new_size, interpolation=cv2.INTER_CUBIC)
        return scaled_matrix

    def show_features(self, cols_number=default_hist_param, dft_mat_size=default_dft_param, 
                    dct_mat_size=default_dct_param, window_height=default_grad_param, 
                    scale=default_scale_param):

        fig, [[ax1, ax2, ax3], [ax4, ax5, _]] = plt.subplots(2, 3, figsize=(15, 15))
        
        #print(ax)
        hist = self.get_histogram(cols_number)
        ax1.bar(range(len(hist)), hist)
        ax1.set_xlabel('columns')
        ax1.set_ylabel('frequency')
        ticks = len(ax1.get_xticks())
        ax1.set_xticklabels([str(int(x)) for x in list(np.linspace(0, 255, cols_number)[::(cols_number // ticks)])])
        ax1.set_title(f'Histogram with {cols_number} columns')

        dft_matrix = self.get_dft(dft_mat_size)
        ax2.imshow(dft_matrix, cmap='gray')
        ax2.set_title(f'DFT with {dft_mat_size} matrix size')

        dct_matrix = self.get_dct(dct_mat_size)
        ax3.imshow(dct_matrix, cmap='gray')
        ax3.set_title(f'DCT with {dct_mat_size} matrix size')

        gradient = self.get_gradient(window_height)
        ax4.plot(range(1, len(gradient)+1), gradient)
        ax4.set_xlabel('window position')
        ax4.set_ylabel('difference')
        ax4.set_title(f'Gradient with {window_height} window height')

        scaled_matrix = self.get_scale(scale)
        ax5.imshow(scaled_matrix, cmap='gray')
        ax5.set_title(f'Scale at {scale}% of original size')

        _.axis('off')
        if self.label_number is None:
            title = f'Image №{self.order_number}'
        else:
            title = f'Image №{self.order_number} of person {self.label_number}'
        fig.suptitle(title, fontsize=20)
        plt.show()

        
def extract_features(image, method, param=None):
    if not isinstance(image, Image):
        print('Invalid arguments')
        return
    features = eval(f'image.{method}')
    if param is None:
        return features()
    else:   
        return features(param) 

def get_distance(feature_1, feature_2):
    return cv2.norm(feature_1, feature_2)
    



orl_dir = os.path.join('.', 'ORL Face Database')

if __name__ == "__main__":
    img_path_1 = os.path.join(orl_dir, 's23', '4.pgm')
    # example_hist, _ = get_histogram(img)
    '''
    print(get_dct(img))
    print(get_dft(img))
    print(get_histogram(img))
    print(get_gradient(img))
    print(get_scale(img))
    '''
    img_path_2 = os.path.join(orl_dir, 's24', '5.pgm')
    img_1 = Image(img_path_1, test=True)
    img_2 = Image(img_path_2, test=True)
    #print(img_1.__dict__['get_histogram'])
    print(get_distance(img_1, img_2, 'get_histogram', 30))
    img_1.show_features()
    #plt.imshow(get_scale(img), cmap='gray')
    #plt.show()
    