"""GPU linear probe shared by the representation-probe scripts.

Same objective as sklearn's
    make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs"))
(standardize with train statistics, multinomial logistic regression, L2 with
C=1, unpenalized intercept), solved with jaxopt.LBFGS. All splits are fit at
once with vmap. Verified to match sklearn at convergence.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np


@partial(jax.jit, static_argnames="num_classes")
def _probe_accuracies(x, labels, train_masks, test_masks, num_classes):
    targets = jax.nn.one_hot(labels, num_classes)

    def one_split(train_w, test_w):
        n = train_w.sum()
        mean = (x * train_w[:, None]).sum(0) / n
        var = (jnp.square(x - mean) * train_w[:, None]).sum(0) / n
        z = (x - mean) / jnp.where(var > 0, jnp.sqrt(var), 1.0)

        def loss(params):
            weights, bias = params
            log_probs = jax.nn.log_softmax(z @ weights + bias)
            cross_entropy = -(targets * log_probs).sum(-1)
            # sklearn: C * sum(CE) + 0.5 * ||W||^2, rescaled by 1/n (same optimum).
            return (train_w * cross_entropy).sum() / n + 0.5 * jnp.square(weights).sum() / n

        init = (jnp.zeros((x.shape[1], num_classes)), jnp.zeros(num_classes))
        weights, bias = jaxopt.LBFGS(fun=loss, maxiter=3000, tol=1e-6).run(init).params
        correct = jnp.argmax(z @ weights + bias, axis=-1) == labels
        return (correct * test_w).sum() / test_w.sum()

    return jax.vmap(one_split)(train_masks, test_masks)


def probe_accuracies(x, labels, train_masks, test_masks, num_classes):
    """Test accuracy per split. Masks: [num_splits, N] boolean."""
    return np.asarray(_probe_accuracies(
        jnp.asarray(x, jnp.float32),
        jnp.asarray(labels),
        jnp.asarray(np.asarray(train_masks, np.float32)),
        jnp.asarray(np.asarray(test_masks, np.float32)),
        num_classes,
    ))
