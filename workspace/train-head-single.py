# train_head.py

import os
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import optax
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1) Load precomputed data
pooled = np.load("./cache/pooled_embeddings.npy")  # shape (N, D)
y_all  = np.load("./cache/labels.npy")
with open("./cache/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    pooled, y_all, test_size=0.2, random_state=42
)

# 3) Tthe head
def head_fn(x: jnp.ndarray) -> jnp.ndarray:
    # x: [B, D] → logits [B, num_classes]
    return hk.Linear(num_classes)(x)

head_transformed = hk.transform(head_fn)

rng = jax.random.PRNGKey(42)
dummy = jnp.zeros((1, pooled.shape[1]), dtype=jnp.float32)
head_params = head_transformed.init(rng, dummy)

optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(head_params)

@jax.jit
def update_step(params, opt_state, x_batch, y_batch):
    def loss_fn(p):
        logits = head_transformed.apply(p, rng, x_batch)
        onehot = jax.nn.one_hot(y_batch, num_classes)
        return optax.softmax_cross_entropy(logits, onehot).mean()

    grads = jax.grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    # return new_params, new_opt_state
    return new_params, new_opt_state, loss_fn(params)

# training loop
batch_size = 32
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
for epoch in range(1, 51):
    perm = np.random.permutation(len(X_train))
    X_shuf, Y_shuf = X_train[perm], y_train[perm]

    epoch_loss = 0.0
    epoch_acc  = 0.0

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/5", ncols=80)
    for i in pbar:
        start = i * batch_size
        end   = start + batch_size
        x_batch = jnp.array(X_shuf[start:end], dtype=jnp.float32)
        y_batch = Y_shuf[start:end]
        # head_params, opt_state = update_step(head_params, opt_state, x_batch, y_batch)
        head_params, opt_state, loss = update_step(head_params, opt_state, x_batch, y_batch)
        logits = head_transformed.apply(head_params, rng, x_batch)
        acc    = (jnp.argmax(logits, -1) == y_batch).mean()

        epoch_loss += loss
        epoch_acc  += acc

    avg_loss = epoch_loss / steps_per_epoch
    avg_acc  = (epoch_acc  / steps_per_epoch) * 100
    logits = head_transformed.apply(head_params, rng, jnp.array(X_train, dtype=jnp.float32))
    train_acc = (jnp.argmax(logits, -1) == y_train).mean() * 100
    # pbar.set_postfix(train_acc=f"{train_acc:.1f}%")
    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

logits = head_transformed.apply(head_params, rng, jnp.array(X_test, dtype=jnp.float32))
# test_acc = (jnp.argmax(logits, -1) == y_test).mean() * 100
test_acc = (jnp.argmax(logits, -1) == y_test).mean() * 100
print(f"Test accuracy: {test_acc:.2f}%")
# print(f"Test accuracy: {test_acc:.1f}%")

# sve the head parameters
os.makedirs("cache", exist_ok=True)
with open("./cache/head_params.pkl", "wb") as f:
    cpu_params = jax.tree_util.tree_map(lambda x: np.array(x), head_params)
    pickle.dump(cpu_params, f)
print("Saved head to ./cache/head_params.pkl")
