# precompute_embeddings.py

import os
import pandas as pd
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import pickle
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm

parameters, forward_fn_raw, tokenizer, config = get_pretrained_model(
    model_name="50M_multi_species_v2",
    embeddings_layers_to_save=(20,),
    max_positions=7000,
)
forward_fn = hk.transform(forward_fn_raw)


df = pd.read_csv("./genomic_species.csv")
seqs   = df["sequence"].tolist()
labels = df["genus"].tolist()    


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(labels)
os.makedirs("cache", exist_ok=True)
with open("./cache/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


rng = jax.random.PRNGKey(0)
pooled = []

for seq in tqdm(seqs, desc="Embedding sequences"):
    # tokenize
    token_ids = tokenizer.batch_tokenize([seq])[0][1]
    tokens = jnp.array(token_ids, dtype=jnp.int32)[None, :]
    emb = forward_fn.apply(parameters, rng, tokens)["logits"]  
    pooled.append(np.array(emb.mean(axis=1)[0]))                    

pooled = np.stack(pooled)  # shape (N, D)
np.save("./cache/pooled_embeddings.npy", pooled)
np.save("./cache/labels.npy", y)

print("Saved embeddings → ./cache/pooled_embeddings.npy")
print("Saved labels     → ./cache/labels.npy")