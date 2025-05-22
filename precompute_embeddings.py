# # precompute_embeddings.py

# import os
# import pandas as pd
# import numpy as np
# import jax, jax.numpy as jnp
# import haiku as hk
# import pickle
# from nucleotide_transformer.pretrained import get_pretrained_model
# from tqdm import tqdm


# @jax.pmap
# def embed_batch(params, toks, rng):
#     emb = forward_fn.apply(params, rng, toks)
#     return emb["embeddings_20"]


# n_dev = jax.device_count()        # should be 2
# per_dev = 8                       # e.g. 8 seqs per GPU
# batch_size = n_dev * per_dev

# # 1) Load & freeze the backbone
# parameters, forward_fn_raw, tokenizer, config = get_pretrained_model(
#     model_name="500M_multi_species_v2",
#     embeddings_layers_to_save=(20,),
#     max_positions=7000,
# )
# forward_fn = hk.transform(forward_fn_raw)

# # 2) Read your CSV
# df = pd.read_csv("./genomic_species.csv")
# seqs   = df["sequence"].tolist()
# labels = df["genus"].tolist()     # change to "species_epithet" if you prefer

# # 3) Encode labels
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y  = le.fit_transform(labels)
# os.makedirs("cache", exist_ok=True)
# with open("./cache/label_encoder.pkl", "wb") as f:
#     pickle.dump(le, f)

# # 4) Compute & save pooled embeddings
# rng = jax.random.PRNGKey(0)
# pooled = []

# for seq in tqdm(seqs, desc="Embedding sequences"):
#     # tokenize
#     token_ids = tokenizer.batch_tokenize([seq])[0][1]
#     tokens = jnp.array(token_ids, dtype=jnp.int32)[None, :]
#     # forward → take layer-20 & mean-pool
#     emb = forward_fn.apply(parameters, rng, tokens)["embeddings_20"]   # [1, L, D]
#     pooled.append(np.array(emb.mean(axis=1)[0]))                     # → [D]

# pooled = np.stack(pooled)  # shape (N, D)
# np.save("./cache/pooled_embeddings.npy", pooled)
# np.save("./cache/labels.npy", y)

# print("Saved embeddings → ./cache/pooled_embeddings.npy")
# print("Saved labels     → ./cache/labels.npy")


# precompute_embeddings.py

import os
# throttle JAX’s pre-allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
from functools import partial
import pandas as pd
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import pickle
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# load & freeze backbone
parameters, forward_fn_raw, tokenizer, config = get_pretrained_model(
    model_name="500M_multi_species_v2",
    embeddings_layers_to_save=(20,),
    max_positions=7000,
)
forward_fn = hk.transform(forward_fn_raw)

# 
@partial(jax.pmap, in_axes=(None, 0, 0))
def embed_batch(params, toks, rng):
    out = forward_fn.apply(params, rng, toks)
    return out["embeddings_20"]   

df     = pd.read_csv("./genomic_species.csv")
seqs   = df["sequence"].tolist()
labels = df["genus"].tolist()
le     = LabelEncoder()
y      = le.fit_transform(labels)
os.makedirs("cache", exist_ok=True)
with open("cache/label_encoder.pkl","wb") as f:
    pickle.dump(le, f)

# Shard 
n_dev    = jax.device_count()    # e.g. 2
per_dev  = 8                     # how many seqs per GPU
batch_sz = n_dev * per_dev
max_pos  = config.max_positions


rng    = jax.random.PRNGKey(0)
pooled = []

for i in tqdm(range(0, len(seqs), batch_sz), desc="Embedding"):
    chunk = seqs[i : i + batch_sz]
    orig  = len(chunk)
    # sequence padding
    if orig < batch_sz:
        chunk += [chunk[-1]] * (batch_sz - orig)


    token_ids = [tokenizer.batch_tokenize([s])[0][1] for s in chunk]
    toks = np.stack([t[:max_pos] for t in token_ids])
    toks = toks.reshape((n_dev, per_dev, -1))
    toks = jnp.array(toks, dtype=jnp.int32)


    rng, *sub = jax.random.split(rng, n_dev+1)
    rngs = jnp.stack(sub)   


    embs_sharded = embed_batch(parameters, toks, rngs)


    embs_sharded = jnp.mean(embs_sharded, axis=2)
    flat = np.array(embs_sharded).reshape((batch_sz, -1))

    pooled.extend(flat[:orig])

# 6) save
pooled = np.stack(pooled)
np.save("cache/pooled_embeddings.npy", pooled)
np.save("cache/labels.npy", y)
print("Done. Saved to cache/.")
