import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cur_dir= os.getcwd()
base_dir=os.path.join(cur_dir,"scgnn_lambda")


type_name="type3"
dataset_name="ms"



def load_and_plot_results(base_dir, dataset_name, type_name):
    path_list= ["GG-CG","GC-CG","CG-CC","CC-CC"]
    lambda_list = list(np.flip(np.array([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0])))
  
    #plt.figure(figsize=(6,6))
    
    acc_list= np.zeros((len(lambda_list),len(path_list)))
    std_list= np.zeros((len(lambda_list),len(path_list)))
    
    for i,lm in enumerate(lambda_list):
        lm= f"lamda_{str(lm)}"
        for j, pt in enumerate(path_list):
                    
            load_path = os.path.join(base_dir, dataset_name,type_name,lm,pt)
            try:
                final_results_file = os.path.join(load_path, os.listdir(load_path)[-1])
            except IndexError:
                continue  # Skip if no files found in the directory

            with open(final_results_file, "rb") as f:
                loaded_results = pickle.load(f)

            acc_list[i,j]= np.mean(np.array(loaded_results['test_acc']))
            std_list[i,j]= np.std(np.array(loaded_results['test_acc']))
    
        # Number of columns
    num_columns = acc_list.shape[1]

    # Different format strings for each column
    formats = ['-o', '--s', '-.^', ':d']

    # Plot each column
    for i in range(num_columns):
        plt.errorbar(range(len(acc_list)), acc_list[:, i], yerr=std_list[:, i], fmt=formats[i], capsize=5, label=path_list[i])
    
    
    # Bold the frame
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the desired linewidth

     
    plt.grid(True)
    # Adjust layout to prevent overlap
  
    plt.legend(path_list)
    # Set the x-ticks
    plt.xticks(ticks=range(len(lambda_list)), labels=lambda_list,fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.xlabel(r'Lambda($\lambda$)',fontsize=15,labelpad=1,fontweight="bold")
    plt.title(dataset_name.upper(),fontweight="bold")
    # Bold the legend labels
    plt.legend(prop={'weight': 'bold'})
    plt.ylabel(r'Test Accuracy ($\%$)',fontsize=15,labelpad=1, fontweight="bold")
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"./lambda_plot/{dataset_name}_acc.png")

load_and_plot_results(base_dir, dataset_name, type_name)