from pathlib import Path
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


dataset_name="pancreas"

print(dataset_name)

##################################################################################
if dataset_name == "ms":
    data_dir = Path("../data/ms")
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])

if dataset_name == "pancreas": #RB
    data_dir = Path("../data/pancreas")
    adata = sc.read(data_dir / "demo_train.h5ad")
    adata_test = sc.read(data_dir / "demo_test.h5ad")
    adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"    
    data_is_raw = False
    filter_gene_by_counts = False   
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])

if dataset_name == "myeloid":
    data_dir = Path("../data/mye/")
    adata = sc.read(data_dir / "reference_adata.h5ad")
    adata_test = sc.read(data_dir / "query_adata.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata_test_raw = adata_test.copy()
    data_is_raw = False
    filter_gene_by_counts = False   
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])
##################################################################################


#### Take results from the save transformer model
file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
with open(file_path, "rb") as file:
        results= pickle.load(file)   
seed_list=results["seed_numbers"]


# We can automatize this, but I just want some visual examples
path_to_plot= "/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/scgnn_merged/pancreas/type3/GC-CG/dname_pancreas_path_[GC-CG]_type_type3_seedid_4_seed_16"

with open(path_to_plot, "rb") as f:
    loaded_results = pickle.load(f)

y_test_preds= loaded_results["test_preds"][-1]



# This part will be used for further plotting, it can be used in a different file, no problem at all
id2type=results["id_maps"]           


print(id2type)

print(adata_test_raw.obs["celltype"].unique())




adata_test_raw.obs["predictions"]=[id2type[p]  for p in  y_test_preds]


palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_= sns.color_palette("Set2",n_colors=14)
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                  "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
                  "#98df8a", "#ff9896"]

palette_ = {c: custom_palette[i] for i, c in enumerate(adata.obs["celltype"].unique())}
print(palette_)
#print(adata_test_raw.to_df())


 
plt.figure(figsize=(8,8))        
with plt.rc_context({"figure.figsize": (3,3), "figure.dpi": (300),"axes.labelsize": 8, "axes.linewidth": 0.75}):
    sc.pl.umap(
        adata_test_raw,
        color="celltype",
        palette=palette_,
        show=False,
        legend_fontsize=3,
        legend_loc="right margin",
        size=8,
        title=""
        
    )

if dataset_name=="ms":
    fig_label="MS"
else:
    fig_label=dataset_name.capitalize()
    
plt.xlabel("Annotated")

plt.ylabel(fig_label)
plt.savefig("results_annotated.png", dpi=300)


with plt.rc_context({"figure.figsize": (3,3), "figure.dpi": (300),"axes.labelsize": 8, "axes.linewidth": 0.75}):
        sc.pl.umap(
            adata_test_raw,
            color="predictions",
            palette=palette_,
            show=False,
            legend_fontsize=3,
            legend_loc="right margin",
            size=8,
          
        
        )
        

if dataset_name=="ms":
    fig_label="MS"
else:
    fig_label=dataset_name.capitalize()
    
plt.xlabel("Predicted")
plt.ylabel(fig_label)

plt.savefig("results_predicted.png", dpi=300)

