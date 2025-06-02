import sys
from pathlib import Path

# Get the parent directory of the notebook (i.e., the 'project/' folder)
parent_dir = Path().resolve().parent
sys.path.append(str(parent_dir))

from embedding_quality import embedding_quality
import numpy as np
import pickle
import openTSNE
from openTSNE import TSNE
import torchvision
from sklearn.decomposition import PCA
import pandas as pd


number_rs = 4
lambdas_list = np.linspace(0, 1, 41)

print('------------------------- Retina -------------------------')

# Retina
data = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/retina/3000_no_std_pca50.npy')
labels = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/retina/labels 1.npy')

retina_pca2 = data[:, :2]
init = retina_pca2 / retina_pca2[:,0].std()

results_dict = {}
for seed in range(number_rs):
    seed_key = f"seed_{seed}"
    results_dict[seed_key] = {}

    for i, l in enumerate(lambdas_list):
        print(f'Running {i+1}/{len(lambdas_list) * number_rs} with seed {seed} and lambda {l}')
        embedder = TSNE(initialization=init, regularization=True, reg_lambda=l, reg_embedding=init, reg_scaling='norm', reg_scaling_dims='one')
        embd = embedder.fit(data)
        eval = embedding_quality(embd, data, labels)

        l_key = f"lambda_{l}"
        results_dict[seed_key][l_key] = {
            'embedding': np.array(embd),
            'eval': eval
        }

with open('/gpfs01/berens/user/nkury/tsne_pca/openTSNE/results/retina_results_opentsne_pca_reg.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

print('------------------------- Zebrafish -------------------------')
# Zebrafish
data = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/zfish/zfish.data.npy')
labels = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/zfish/zfish.labels.npy')
# zfish_alt_colors = np.load('data/zfish/zfish.altlabels.npy')

zfish_pca2 = data[:, :2]
init = zfish_pca2 / zfish_pca2[:,0].std()

results_dict = {}
for seed in range(number_rs):
    seed_key = f"seed_{seed}"
    results_dict[seed_key] = {}

    for i, l in enumerate(lambdas_list):
        print(f'Running {i+1}/{len(lambdas_list) * number_rs} with seed {seed} and lambda {l}')
        embedder = TSNE(initialization=init, regularization=True, reg_lambda=l, reg_embedding=init, reg_scaling='norm', reg_scaling_dims='one')
        embd = embedder.fit(data)
        eval = embedding_quality(embd, data, labels)

        l_key = f"lambda_{l}"
        results_dict[seed_key][l_key] = {
            'embedding': np.array(embd),
            'eval': eval
        }

with open('/gpfs01/berens/user/nkury/tsne_pca/openTSNE/results/zfish_results_opentsne_pca_reg.pkl', 'wb') as f:
    pickle.dump(results_dict, f)

print('------------------------- C. elegans -------------------------')
# C. elegans

data = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/c_elegans/c_elegans_50pc.npy')
labels = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/c_elegans/c_el_cell_types.npy', allow_pickle=True).astype(str)

c_el_pca2 = data[:, :2]
init = c_el_pca2 / c_el_pca2[:,0].std()

results_dict = {}
for seed in range(number_rs):
    seed_key = f"seed_{seed}"
    results_dict[seed_key] = {}

    for i, l in enumerate(lambdas_list):
        print(f'Running {i+1}/{len(lambdas_list) * number_rs} with seed {seed} and lambda {l}')
        embedder = TSNE(initialization=init, regularization=True, reg_lambda=l, reg_embedding=init, reg_scaling='norm', reg_scaling_dims='one')
        embd = embedder.fit(data)
        eval = embedding_quality(embd, data, labels)

        l_key = f"lambda_{l}"
        results_dict[seed_key][l_key] = {
            'embedding': np.array(embd),
            'eval': eval
        }

with open('/gpfs01/berens/user/nkury/tsne_pca/openTSNE/results/c_elegans_results_opentsne_pca_reg.pkl', 'wb') as f:
    pickle.dump(results_dict, f)