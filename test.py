import pandas as pd
import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
from tqdm import tqdm

# 1) Load model
parameters, forward_fn_raw, tokenizer, config = get_pretrained_model(
    model_name="250M_multi_species_v2",
    embeddings_layers_to_save=(20,),
    max_positions=5012,
)
forward_fn = hk.transform(forward_fn_raw)


@jax.jit
def apply_model(params, rng, tokens):
    return forward_fn.apply(params, rng, tokens)

def pick_device():
    devs = jax.devices()
    # filter by d.platform == "gpu"
    for d in devs:
        if d.platform == "gpu":
            return d
    # no GPU found → return first (CPU)
    return devs[0]

# embedding fn
def get_embedding(sequences):
    # tokenize
    token_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(token_ids, dtype=jnp.int32)

    device = pick_device()
    tokens = jax.device_put(tokens, device)

    # run model
    rng = jax.random.PRNGKey(0)
    outs = apply_model(parameters, rng, tokens)
    print(outs)
    print(outs.keys())

    return outs["embedding_20"]

if __name__ == "__main__":
    df = pd.read_csv("genomic_species.csv")
    sample = list(df["sequence"])[:2]
    embs = get_embedding(sample)
    print("Device used:", pick_device())
    print("Embeddings shape:", embs.shape)
