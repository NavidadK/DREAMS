import sys
from pathlib import Path

# Get the parent directory of the notebook (i.e., the 'project/' folder)
parent_dir = Path().resolve().parent
sys.path.append(str(parent_dir))

from embedding_quality import embedding_quality

import openTSNE
print(openTSNE.__file__)
from openTSNE import TSNE
import pickle
import numpy as np

print("Imports completed successfully.")

tasic_pca50 = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/tasic/tasic-pca50.npy')
tasic_ttypes = np.load('/gpfs01/berens/user/nkury/tsne_pca/data/tasic/tasic-ttypes.npy')

tasic_pca2 = tasic_pca50[:, :2]
tasic_pca2_scaled = tasic_pca2 / tasic_pca2[:,0].std()

exag_vals = [350, 500, 750, 1000, 1500, 2000, 3000, 5000]

results_dict = {}
for seed in range(1):
    seed_key = f"seed_{seed}"
    results_dict[seed_key] = {}

    for i, exag in enumerate(exag_vals):
        print(f'Running {i+1}/{len(exag_vals)}')
        embedder = TSNE(initialization=tasic_pca2_scaled, exaggeration=exag)
        embd = embedder.fit(tasic_pca50)
        eval = embedding_quality(embd, tasic_pca50, tasic_ttypes)

        exag_key = f"exag_{exag}"
        results_dict[seed_key][exag_key] = {
            'embedding': np.array(embd),
            'eval': eval
        }

with open('/gpfs01/berens/user/nkury/tsne_pca/openTSNE/results/tasic_results_openTSNE_exag_high.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
