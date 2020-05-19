from image_process import Image 
from image_process import extract_features, get_distance
from image_process import orl_dir
from image_process import histogram, dft, dct, gradient, scale
import numpy as np 
from matplotlib import pyplot as plt, gridspec as gspec 
from matplotlib.animation import FuncAnimation 
from sklearn.model_selection import KFold, train_test_split
import os

fig_dir = os.path.join('.', 'figures', 'task_3') 
all_methods = [histogram, dft, dct, gradient, scale]
opt_params = [21, 9, 5, 10, 36]
'''
def _predict_by_images(X_train, X_test, method, param=None):
    predicted_labels = np.zeros(len(X_test))
    #print(type(X_test))
    
    if not isinstance(X_test, list) or not isinstance(X_test, tuple) or not isinstance(X_test, np.ndarray):
        #print('*')
        X_test = [X_test]
    
    for i, tested_image in enumerate(X_test):
       
        test_features = extract_features(tested_image, method, param)
        dists = np.zeros(len(X_train))
        for j, reference_image in enumerate(X_train):
            reference_features = extract_features(reference_image, method, param)
            dists[j] = get_distance(test_features, reference_features)
        bestind = np.argmin(dists)
        predicted_value = X_train[bestind].label_number
        predicted_labels[i] = predicted_value
    
    return predicted_labels

def calculate_accuracy(real_labels, predicted_labels):
    #print(predicted_labels)
    #print(real_labels)
    result = np.mean(predicted_labels == real_labels)
    #print(result)
    return result


def fixed_train_size_predict(X, method, param=None, train_size_=4, people_number=40):
    X_train = []
    X_test = []
    #features_train = []
    #features_test = []
    for i in range(1, people_number+1):
        cur_X = np.array([image for image in X if image.label_number == i])
        #cur_features = extr_features(cur_X, method, param)
        #cur_X_train, cur_X_test, cur_features_train, cur_features_test = train_test_split(cur_X, cur_features, train_size=train_size_)
        #print(cur_X_test)
        cur_X_train, cur_X_test = train_test_split(cur_X, train_size=train_size_)
        X_train.extend(list(cur_X_train))
        X_test.extend(list(cur_X_test))
        #features_train.extend(list(cur_features_train))
        #features_test.extend(list(cur_features_test))
    
    accuracy = predict_by_features(*list_of_sets)[1]
    return accuracy
'''
def load_database(dir=orl_dir, people_number=40, images_number=10):
    if people_number > 40:
        people_number = 40
    if images_number > 10:
        images_number = 10
    images_list = []
    for i in range(1, people_number+1):
        for j in range(1, images_number+1):
            image_path = os.path.join(dir, f's{i}', f'{j}.pgm')
            images_list.append(Image(image_path, test=False))

    return np.array(images_list)


def extr_features(X, method, param):
    features_array = []
    for image in X:
        feature = extract_features(image, method, param)
        features_array.append(feature)
    features_array = np.array(features_array)
    return features_array


def predict_by_features(X_train, X_test, features_train, features_test):
    predicted_labels = np.zeros(len(X_test))
    real_labels = np.zeros(len(X_test))
    best_matches = []
    for i, tested_feature in enumerate(features_test):
        dists = np.zeros(len(X_train))
        for j, reference_feature in enumerate(features_train):
            dists[j] = get_distance(tested_feature, reference_feature)
        bestind = np.argmin(dists)
        best_match = X_train[bestind]
        predicted_value = int(best_match.label_number)
        best_matches.append(best_match)
        predicted_labels[i] = predicted_value
        real_labels[i] = int(X_test[i].label_number)
        
    accuracy = np.mean(predicted_labels == real_labels)
    #print(accuracy)
    return predicted_labels, accuracy, X_train[bestind]


def predict(X_train, X_test, method, param=None):
    features_train = extr_features(X_train, method, param)

    features_test = extr_features(X_test, method, param)
    predicted_labels, accuracy, best_ = predict_by_features(X_train, X_test, features_train, features_test)
    return predicted_labels, accuracy, best_


def cross_validation_test(X, method, param=None, nsplits=10):
    features_array = extr_features(X, method, param)
    kf = KFold(n_splits=nsplits, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(X):
        #print('*')
        X_train, X_test = X[train_index], X[test_index]
        features_train, features_test = features_array[train_index], features_array[test_index]
        curr_acc = predict_by_features(X_train, X_test, features_train, features_test)[1]
        accuracies.append(curr_acc)
    
    total_acc = np.mean(accuracies)
    return total_acc


def multiple_cross_validation(X, method, param=None, nsplits=10, ntimes=2):
    accuracies = []
    for i in range(ntimes):
        acc = cross_validation_test(X, method, param, nsplits)
        accuracies.append(acc)
    total_acc = np.mean(accuracies)
    return total_acc



def get_method_strdata(method):
    if method == dft or method == dct:
        title = method.replace('get_', '').upper()
        xlabel = 'Matrix size'
    elif method == gradient:
        title = method.replace('get_g', 'G')
        xlabel = 'Window height'
    elif method == histogram:
        title = method.replace('get_h', 'H')
        xlabel = 'Number of columns'
    elif method == scale:
        title = method.replace('get_s', 'S')
        xlabel = 'Percent of original size'
    
    return title, xlabel
        

def vary_param(X, method, cvsplits=10, cvtimes=2, step=1, dscr=0.005, plot=True, savefig=False, filename='vary_param'):
    min_param, max_param = get_min_max(method)
    param_range = range(min_param, max_param+1, step)
    accuracies = np.zeros(len(param_range))
    for i, param in enumerate(param_range):
        accuracies[i] = multiple_cross_validation(X, method, param, cvsplits, cvtimes)
        print(f'{method}: {param}')
    max_value = np.max(accuracies)
    if method != gradient:
        best_step = np.min(np.nonzero(max_value - accuracies <= dscr))
    else:
        best_step = np.max(np.nonzero(max_value - accuracies <= dscr))
    best_param = param_range[best_step]
    #print(best_param)
    #print(accuracies[best_step])
    if plot:
        plt.cla()
        title, xlabel = get_method_strdata(method)
        plt.plot(param_range, accuracies)
        plt.plot(param_range, np.repeat(max_value, len(param_range)), '--')
        plt.title(title)
        plt.grid(True)
        plt.xlabel(xlabel)
        param_range = list(param_range)
        while len(param_range) > 12:
            param_range = (param_range[::-2])[::-1]
        plt.xticks(param_range)
        plt.ylabel('Accuracy')
        if savefig:
            plt.savefig(filename)
        else:  
            plt.show()
    return best_param


def split_fixed_size(X, train_size_=5, people_number=40):
    X_train = []
    X_test = []
    for i in range(1, people_number+1):
        cur_X = np.array([image for image in X if image.label_number == i])
        cur_X_train, cur_X_test = train_test_split(cur_X, train_size=train_size_)
        X_train.extend(list(cur_X_train))
        X_test.extend(list(cur_X_test))
    list_of_sets = [np.array(st) for st in [X_train, X_test]]
    return tuple(list_of_sets)


def calculate_cumulative_accuracy(X, method, param=None, train_size_=5, savefig=False, filename='cumulative'):
    X_train, X_test = split_fixed_size(X, train_size_)
    
    accuracies = []
    cumulatives = []
    max_images = len(X_test)
    features_train = extr_features(X_train, method, param)
    for cur_image in X_test:
        cur_wrapper = np.array([cur_image])
        cur_features = extr_features(cur_wrapper, method, param)
        cur_accuracy = predict_by_features(X_train, cur_wrapper, features_train, cur_features)[1]
        accuracies.append(cur_accuracy)
        cur_cumulative = np.mean(accuracies)
        cumulatives.append(cur_cumulative)
    title, param_name = get_method_strdata(method)
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.xlabel('Size of test set')
    plt.ylabel('Accuracy, cumulative')
    plt.xticks(list(range(0, max_images+1, 20)))
    plt.yticks(np.append(np.arange(0, 0.81, 0.2), np.arange(0.9, 1.01, 0.01)), fontsize=8)
    if param is None:
        param = 'default'
    plt.title(f'{title}, {param_name} = {param}')
    plt.plot(range(1, max_images+1), cumulatives)
    if savefig:
        plt.savefig(filename)
    else:  
        plt.show()
    

def calculate_cumulative_voting(X, methods=all_methods, params=None, train_size_=5, savefig=False, filename='cumulative_voting'):
    if params is None:
        params = [None]*len(methods)
    X_train, X_test = split_fixed_size(X, train_size_)
    accuracies = []
    cumulatives = []
    max_images = len(X_test)
    
    features_train_list = []
    for i, cur_image in enumerate(X_test):
        for j, method in enumerate(methods):
            if i == 0:
                features_train_list.append(extr_features(X_train, method, params[j]))
            cur_wrapper = np.array([cur_image])
            cur_features = extr_features(cur_wrapper, method, params[j])
            if j == 0:
                predicted_matrix = predict_by_features(X_train, cur_wrapper, features_train_list[j], cur_features)[0]
            else:
                predicted_matrix = np.vstack((predicted_matrix, predict_by_features(X_train, cur_wrapper, features_train_list[j], cur_features)[0]))
        predicted_matrix = np.transpose(predicted_matrix)
        predicted_label = []
        for row in predicted_matrix:
            u, indices = np.unique(row, return_inverse=True)
            most_frequent = u[np.argmax(np.bincount(indices))]
            predicted_label.append(most_frequent)
        predicted_label = np.array(predicted_label)
        real_label = int(cur_image.label_number)
        cur_accuracy = int(real_label == predicted_label)
        accuracies.append(cur_accuracy)
        cur_cumulative = np.mean(accuracies)
        cumulatives.append(cur_cumulative)
 
    title = 'Parallel classifier'
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.xlabel('Size of test set')
    plt.ylabel('Accuracy, cumulative')
    plt.xticks(list(range(0, max_images+1, 20)))
    plt.yticks(np.append(np.arange(0, 0.81, 0.2), np.arange(0.9, 1.01, 0.01)), fontsize=8)
    plt.title(f'{title}')
    plt.plot(range(1, max_images+1), cumulatives)
    if savefig:
        plt.savefig(filename)
    else:  
        plt.show()



def vote_predict(X_train, X_test, methods=all_methods, params=None, same_train_=False):
    if params is None:
        params = [None]*len(methods)
    for i, method in enumerate(methods):
        if same_train_:
            features_train_ = extr_features(X_train, method, params[i])
        if i == 0:
            predicted_matrix = predict(X_train, X_test, method, params[i])[0]
        else:
            predicted_matrix = np.vstack((predicted_matrix, predict(X_train, X_test, method, params[i])[0]))
    predicted_matrix = np.transpose(predicted_matrix)
    predicted_labels = []
    for row in predicted_matrix:
        u, indices = np.unique(row, return_inverse=True)
        most_frequent = u[np.argmax(np.bincount(indices))]
        predicted_labels.append(most_frequent)
    predicted_labels = np.array(predicted_labels)
    real_labels = np.zeros(len(X_test))
    for i, image in enumerate(X_test):
        real_labels[i] = int(image.label_number)
    accuracy = np.mean(predicted_labels == real_labels)
    return predicted_labels, accuracy
    

def vary_train_size(X, methods=[histogram, dft, dct, gradient, scale],
                  params=None, voting=False, voting_params=None, savefig=False, filename='vary_train_size'):
    
    if params is None:
        params = [None]*len(methods)
    if voting_params is None:
        voting_params = [None]*5
    if len(methods) == 5 and params[0] is not None:
        voting_params = params
    train_sizes = range(1, 10)
    accuracies = np.zeros(9)
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.xlabel('Size of train set for each class')
    plt.xticks(train_sizes)
    plt.ylabel('Accuracy')
    plt.title('Variable train set size')
    
    for i, method in enumerate(methods):
        title, param_name = get_method_strdata(method)
        for size in train_sizes:
            accuracies[size-1] = predict(*split_fixed_size(X, size), method, params[i])[1]
        if params[i] is None:
            params[i] = 'default'
        plt.plot(train_sizes, accuracies, label=f'{title}, {param_name} = {params[i]}')
        
    if voting:
        for size in train_sizes:
            accuracies[size-1] = vote_predict(*split_fixed_size(X, size), params=voting_params)[1]
        plt.plot(train_sizes, accuracies, label='Parallel classifier')
    plt.legend()
        
    if savefig:
        plt.savefig(filename)
    else:  
        plt.show()


def get_min_max(method):
    if method in [dct, dft, gradient]:
        return 2, 35
    if method == histogram:
        return 5, 80
    if method == scale:
        return 5, 100

def strip_cumulative_accuracy(X_train, X_test, method, param, train_size_):
    pass
    

def real_time_show(train_size_=5, main_method=dft, voting=False):
    plt.ion()
    X_train, X_test = split_fixed_size(X, train_size_)
    #fig = plt.figure(0)
    #fig.clf()
    def set_plot():
        
        # large subplot
        
       

        for ax in (fake_ax1, fake_ax2):
            ax.axis('off')
        normal_axes = (img_ax, method_ax1, method_ax2, method_ax3, method_ax4,
        method_ax5, descr_ax1, descr_ax2, descr_ax3, descr_ax4, descr_ax5, result_ax)
        for ax in normal_axes:
            if ax != result_ax:
                ax.set_xticks([])
                ax.set_yticks([])

    fig = plt.figure(0, figsize=(8, 8))
    fig.clf()
    grid = gspec.GridSpec(4, 8)
    img_ax = fig.add_subplot(grid[0:2, 0:2])
    method_ax1 = fig.add_subplot(grid[0, 2])
    method_ax2 = fig.add_subplot(grid[1, 2])
    method_ax3 = fig.add_subplot(grid[0, 3])
    method_ax4 = fig.add_subplot(grid[1, 3])
    method_ax5 = fig.add_subplot(grid[0, 4])
    fake_ax1 = fig.add_subplot(grid[1, 4])
    descr_ax1 =  fig.add_subplot(grid[0, 5])
    descr_ax2 = fig.add_subplot(grid[1, 5])
    descr_ax3 = fig.add_subplot(grid[0, 6])
    descr_ax4 = fig.add_subplot(grid[1, 6])
    descr_ax5 = fig.add_subplot(grid[0, 7])
    fake_ax2 = fig.add_subplot(grid[1, 7])
    result_ax = fig.add_subplot(grid[2:, :])

    accuracies = []
    cumulatives = []
    cur_number_list = []
    max_images = len(X_test)
    number_range = list(range(1, max_images+1))
    if main_method == dft:
        main_param = opt_params[1]
    elif main_method == histogram:
        main_param = opt_params[0]
    elif main_method == dct:
        main_param = opt_params[2]
    elif main_method == gradient:
        main_param = opt_params[3]
    elif main_method == scale:
        main_param = opt_params[4]
    
    if not voting:
        features_train = extr_features(X_train, main_method, main_param)
        features_train_list = []
    else:
        features_train_list = []
    for j in range(len(X_test)):
        set_plot()
        cur_image = X_test[j]
        

        
        for ax in (fake_ax1, fake_ax2):
            ax.axis('off')
        normal_axes = (img_ax, method_ax1, method_ax2, method_ax3, method_ax4,
        method_ax5, descr_ax1, descr_ax2, descr_ax3, descr_ax4, descr_ax5, result_ax)
        for ax in normal_axes:
            if ax != result_ax:
                ax.set_xticks([])
                ax.set_yticks([])

        if not voting:
            cur_wrapper = np.array([cur_image])
            cur_features = extr_features(cur_wrapper, main_method, main_param)
            labels, cur_accuracy = predict_by_features(X_train, cur_wrapper, features_train, cur_features)[0:2]
            #print(labels)
            accuracies.append(cur_accuracy)
            #print(cur_accuracy)
            cur_cumulative = np.mean(accuracies)
            cumulatives.append(cur_cumulative)
            for i, method in enumerate(all_methods):
                if j == 0:
                    features_train_list.append(extr_features(X_train, method, opt_params[i]))
                cur_features = extr_features(cur_wrapper, method, opt_params[i])
                #predicted_image = predict(X_train, np.array([cur_image]), method, opt_params[i])[2].matrix
                predicted_image = predict_by_features(X_train, cur_wrapper, features_train_list[i], cur_features)[2].matrix
                image_to_show = f'method_ax{i+1}.imshow(predicted_image, cmap="gray")'
                
                #image_to_show = f'method_ax{i+1}.imshow(predict(X_train, np.array([cur_image]), {method}, opt_params[{i}])[2].matrix, cmap="gray")'
                _title = method.replace("get_", "")
                title_to_show = f'method_ax{i+1}.set_title(_title)'
                eval(image_to_show)
                eval(title_to_show)

        else:
            for i, method in enumerate(all_methods):
                if j == 0:
                    features_train_list.append(extr_features(X_train, method, opt_params[i]))
                cur_wrapper = np.array([cur_image])
                cur_features = extr_features(cur_wrapper, method, opt_params[i])
                predicted_data = predict_by_features(X_train, cur_wrapper, features_train_list[i], cur_features)
                if i == 0:
                    predicted_matrix = predicted_data[0]
                else:
                    predicted_matrix = np.vstack((predicted_matrix, predicted_data[0]))
                #predicted_image = predict(X_train, np.array([cur_image]), method, opt_params[i])[2].matrix
                predicted_image = predicted_data[2].matrix
                image_to_show = f'method_ax{i+1}.imshow(predicted_image, cmap="gray")'
                
                #image_to_show = f'method_ax{i+1}.imshow(predict(X_train, np.array([cur_image]), {method}, opt_params[{i}])[2].matrix, cmap="gray")'
                title_to_show = f'method_ax{i+1}.set_title(method.replace("get_", ""))'
                eval(image_to_show)
                eval(title_to_show)
            predicted_matrix = np.transpose(predicted_matrix)
            predicted_label = []
            for row in predicted_matrix:
                u, indices = np.unique(row, return_inverse=True)
                most_frequent = u[np.argmax(np.bincount(indices))]
                predicted_label.append(most_frequent)
            predicted_label = np.array(predicted_label)
            real_label = int(cur_image.label_number)
            cur_accuracy = int(real_label == predicted_label)
            accuracies.append(cur_accuracy)
            cur_cumulative = np.mean(accuracies)
            cumulatives.append(cur_cumulative)

        img_ax.imshow(cur_image.matrix, cmap='gray')
        img_ax.set_title(f'Tested image')
       
        hist = cur_image.get_histogram(opt_params[0])
        descr_ax1.bar(range(len(hist)), hist)
        descr_ax1.set_title(f'Histogram')
        dft_matrix = cur_image.get_dft(opt_params[1])
        descr_ax2.imshow(dft_matrix, cmap='gray')
        descr_ax2.set_title(f'DFT')
        dct_matrix = cur_image.get_dct(opt_params[2])
        descr_ax3.imshow(dct_matrix, cmap='gray')
        descr_ax3.set_title(f'DCT')
        gradient_data = cur_image.get_gradient(opt_params[3])
        descr_ax4.plot(range(1, len(gradient_data)+1), gradient_data)
        descr_ax4.set_title(f'Gradient')
        scaled_matrix = cur_image.get_scale(opt_params[4])
        descr_ax5.imshow(scaled_matrix, cmap='gray')
        descr_ax5.set_title(f'Scale')


        if not voting:
            title, param_name = get_method_strdata(main_method)
            result_ax.set_title(f'{title}, {param_name} = {main_param}')
        else:
            result_ax.set_title(f'Parallel classifier')
        result_ax.set_xlabel('Size of test set')
        result_ax.set_ylabel('Accuracy, cumulative')
        result_ax.set_xticks(list(range(0, max_images+1, 20)))
        result_ax.set_yticks(np.append(np.arange(0, 0.8, 0.2), np.arange(0.8, 1.01, 0.05)))
        result_ax.grid(True)
        
        
        cur_number_list.append(number_range[j])
        #cur_vals_list.append(cumulatives[j])
        result_ax.plot(cur_number_list, cumulatives)
        fig.suptitle('DEMO', fontsize=20)
        #fig.tight_layout()
        plt.draw()
        plt.pause(0.001)
        for ax in normal_axes:
            ax.cla()

    



if __name__ == '__main__':
    X = load_database()
    #print(X)
    
    for i in range(8, 65, 8):
        #print([cross_validation_test(X, 2, histogram, i)])
        #features_array = extr_features(X, histogram, i)
        #print(predict(X, X, histogram, i)[1])
        pass
    '''
    best_params = []   
    for method in all_methods:
        best_params.append(vary_param(X, method, cvsplits=10, cvtimes=2, 
                         dscr=0.003, plot=True, savefig=True, 
                         filename=os.path.join(fig_dir, f'opt_param_{method.replace("get_", "")}')))

    print(best_params)
    '''
    
    #calculate_cumulative_accuracy(X, gradient)
    #X_train, X_test = split_fixed_size(X, train_size_=4)
    #print(vote_predict(X_train, X_test))
    #vary_train_size(X, methods=all_methods, params=opt_params, voting=True)
    #calculate_cumulative_voting(X, params=opt_params)
    real_time_show(main_method=scale, voting=True)
    







        

        


    



        
        
            

    
