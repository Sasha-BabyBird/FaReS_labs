import os
import cv2
from matplotlib import pyplot as plt, gridspec as gspec
import numpy as np

def align_templates(templates_dir, pics_count=3, types_count=3, gathered=True):
    '''
    UNUSED!

    Resize templates of similar type to make sure their height and width align
    Third argument must be minimal of all types count across all pictures.
    Thus, it ignores templates of types not present in all of pictures.
    If gathered=True, all templates must be in the same folder.
    If gathered=False, all templates must be in subfolders named 'pic_{pic_number}/type_{type_number}'
    '''
    def current_path(i=0, j=0):
        if gathered:
            cur_path = os.path.join(templates_dir, f'template_{j*types_count+i+1}')
        else:
            cur_path = os.path.join(templates_dir, f'pic_{j+1}', f'type_{i+1}')
        
        return cur_path


    for i in range(types_count):
        max_height, max_width = 0, 0
        for j in range(pics_count):
            cur_path = current_path(i, j)
            template = cv2.imread(cur_path, 0)
            height, width = template.shape
            max_height, max_width = max(max_height, height), max(max_width, width)
    
    aligned_templates = list()

    for i in range(types_count):
        for j in range(pics_count):
            cur_path = current_path(i, j)
            template = cv2.imread(cur_path, 0)
            dim = (max_width, max_height)
            template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
            aligned_templates.append((template, i+1))
    
    return resized_templates

def resize_template(template, coef):
    '''
    '''
    if isinstance(template, str):
        template = cv2.imread(template, 0)
    elif isinstance(template, list):
        template = template[0]
    height, width = template.shape
    new_height = height*coef
    new_width = width*coef
    while new_height < 1 or new_width < 1:
        new_height *= 1.5
        new_width *= 1.5
    dim = (int(new_width), int(new_height))
    
    return cv2.resize(template, dim, interpolation=cv2.INTER_AREA)


def match_template(photos_paths, template_path, method_names):
    '''
    '''
    
    def detect_face():
        '''
        '''
        class Rectangle():
            def __init__(self, loc, height, width):
                self.topleft_height = loc[0]
                self.topleft_width = loc[1]
                self.height = height
                self.width = width
        
        img = image.copy()
        coefs = np.arange(0.5, 1.5, 0.04)
        img_height, img_width = image.shape
        templ_height, templ_width = resize_template(template_img, 1).shape
        
        while np.mean([img_height / templ_height, img_width / templ_width]) > 4:
            templ_height *= 1.2
            templ_width *= 1.2
            coefs = np.append(coefs, np.array(coefs[-1]*1.1, coefs[-1]*1.2))
        
        while img_height / templ_height < 0.7 or img_width / templ_width < 0.7:
            templ_height *= 0.6
            templ_width *= 0.6
            coefs = coefs * 0.6
        
        #coefs = np.append(coefs, np.arange(1.0, 10.1, 0.1))
        #coefs = [0.1, 0.2, 0.3, 0.5]
        values = np.array([])
        rectangles = np.array([])
        top_values = 20
        top_local_values = 3

        for coef in coefs:

            scaled_template = resize_template(template_img, coef)
            img_height, img_width = image.shape
            height, width = scaled_template.shape
            if height > img_height or width > img_width:
                break
            method = eval(f'cv2.TM_{md}')
            res = cv2.matchTemplate(img, scaled_template, method)
            resfl = res.flatten()
            if 'SQDIFF' in md:
                locs = np.argpartition(resfl, top_local_values)[:top_local_values]
            else:
                locs = np.argpartition(resfl, -top_local_values)[-top_local_values:]
            locs = [(ind // res.shape[1], ind % res.shape[1]) for ind in locs]
            opt_values = res[tuple(zip(*locs))]
            local_rectangles = [Rectangle(loc, height, width) for loc in locs]
            values = np.append(values, opt_values)
            rectangles = np.append(rectangles, local_rectangles)
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if not 'NORMED' in md:
            #unable to choose maximum of all, so just draw every single rectangle
            best_rectangles = rectangles  
        else:
            if 'SQDIFF' in md: 
                best_indices = values.argsort()[:top_values]
            else:
                best_indices = values.argsort()[-top_values:]
        
            best_rectangles = rectangles[best_indices]
        #print(md, values[best_indices], [rect.height for rect in best_rectangles])
        for rectangle in best_rectangles:
            top_left = (rectangle.topleft_width, rectangle.topleft_height)
            bottom_right = (rectangle.topleft_width + rectangle.width, rectangle.topleft_height + rectangle.height)
            cv2.rectangle(img, top_left, bottom_right, 0, 1)
        photos_img.append(img)
        
    def plot_result():
        imgs_count = len(photos_img)
        cols_count = (imgs_count - 1) // 2 + 3
        fig = plt.figure(1, figsize=(12, 7))
        gspec.GridSpec(cols_count,3)
        # large subplot
        plt.subplot2grid((2,cols_count), (0,0), colspan=2, rowspan=2)
        plt.xticks([]) 
        plt.yticks([])
        plt.title(f'{template_name}')
        plt.imshow(template_img, cmap='gray')
        
        for i, img in enumerate(photos_img):
            row = i % 2
            col = i // 2 + 2
            plt.subplot2grid((2,cols_count), (row ,col))
            plt.xticks([]) 
            plt.yticks([])
            plt.title(subheaders[i], fontsize=7)
            plt.imshow(img, cmap='gray')
        fig.suptitle(main_header, fontsize=15, x=0.45, y=1)
        fig.tight_layout()
        

    template_img = cv2.imread(template_path, 0)
    template_name = os.path.split(template_path)[1]
    if not (isinstance(photos_paths, list) or isinstance(photos_paths, tuple)) or len(photos_paths) == 1:
        if not isinstance(method_names, list):
            method_names = [method_names]
        if isinstance(photos_paths, list) or isinstance(photos_paths, tuple):
            photo_path = photos_paths[0]
        else:
            photo_path = photos_paths
        hdlist = photo_path.split('\\')[1:]
        main_header = ''
        for st in hdlist:
            main_header += f'/{st}'
        image = cv2.imread(photo_path, 0)
        subheaders = method_names
        photos_img = []
        for md in method_names:
            detect_face()
            

    else:
        if isinstance(method_names, list) or isinstance(method_names, tuple):
            md = method_names[0]
        else:
            md = method_names
        main_header = md
        shdlists = [photo_path.split('\\')[2:] for photo_path in photos_paths]
        subheaders = []
        for shdlist in shdlists:
            subheader = ''
            for st in shdlist:
                subheader += f'/{st}'
            subheaders.append(subheader)
        photos_img = []
        for photo_path in photos_paths:
            image = cv2.imread(photo_path, 0)
            detect_face()
    plot_result()
    plt.show()
    
    return 

def plot_result(template, img, method_name):
    '''
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(template, cmap = 'gray')
    ax2.imshow(img, cmap = 'gray')
    ax1.set_title('Template') 
    ax2.set_title('Detected Point')
    for ax in (ax1, ax2):
        ax.set_xticks([]) 
        ax.set_yticks([])
    return fig, (ax1, ax2)

photos_dir = os.path.join('.', 'photos')
templates_dir = os.path.join('.', 'templates')
orl_dir = os.path.join('.', 'ORL Face Database')

if __name__ == '__main__':
    
   
    my_template = os.path.join(templates_dir, 'template_9.jpg')
    
    method_names = ['CCOEFF_NORMED', 'CCORR_NORMED',
                   'SQDIFF_NORMED']
    my_photos = os.path.join(photos_dir, f'photo_8.jpg')
    #my_photos = os.path.join(orl_dir, 's5', '3.pgm')
    method_names = 'CCOEFF_NORMED'
    #my_photos = [os.path.join(photos_dir, f'photo_{i+1}.jpg') for i in range(7)]
    my_photos = [os.path.join(orl_dir, 's15', f'{i+1}.pgm') for i in range(10)]
    match_template(my_photos, my_template, method_names)
    


    