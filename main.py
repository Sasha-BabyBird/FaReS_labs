import numpy as np 
from matplotlib import pyplot as plt
from functions import all_methods, vary_param, vary_train_size, load_database, fig_dir
from functions import calculate_cumulative_accuracy, calculate_cumulative_voting
import os

opt_params = [21, 9, 5, 10, 36]

def get_params_identification_data(save=True):
    best_params = []   
    for method in all_methods:
        best_params.append(vary_param(X, method, cvsplits=10, cvtimes=2, 
                         dscr=0.003, plot=True, savefig=save, 
                         filename=os.path.join(fig_dir, f'opt_param_{method.replace("get_", "")}')))

    return best_params


def save_vary_train_size_data(include_voting=True):
    for i, method in enumerate(all_methods):
        vary_train_size(X, methods=[method], params=[opt_params[i]], savefig=True, 
                       filename=os.path.join(fig_dir, f'vary_train_size_{method.replace("get_", "")}'))
    vary_train_size(X, methods=all_methods, params=opt_params, savefig=True,
                       filename=os.path.join(fig_dir, f'vary_train_size_all'))
    if include_voting:
        vary_train_size(X, methods=all_methods, params=opt_params, savefig=True, voting=True,
                       filename=os.path.join(fig_dir, f'vary_train_size_with_voting'))
    return


def save_cumulative_data(train_size=5, include_voting=True):
    for i, method in enumerate(all_methods):
        calculate_cumulative_accuracy(X, method, param=opt_params[i], train_size_=train_size,
                                      savefig=True, 
                                      filename=os.path.join(fig_dir, f'cumulative_data_{method.replace("get_", "")}'))
    if include_voting:
        calculate_cumulative_voting(X, params=opt_params, train_size_=train_size, savefig=True,
                                    filename=os.path.join(fig_dir, f'cumulative_data_voting'))

    return

if __name__ == '__main__':
    X = load_database()
    #get_params_identification_data()
    save_vary_train_size_data(include_voting=True)
    #save_cumulative_data(train_size=5, include_voting=True)