"""Pure split, selection and scoring rules for the prospective follow-up."""
import hashlib
import json
import numpy as np

TASK_NAMES = ['antonym', 'country-capital', 'english-french', 'present-past',
              'person-occupation', 'singular-plural', 'synonym', 'next_item']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def partition(pairs, seed=20260905, n_dev=25, n_test=50):
    groups = {}
    for pair in pairs:
        bucket = groups.setdefault(pair['input'], [])
        if pair not in bucket:
            bucket.append(dict(pair))
    inputs = sorted(groups)
    if len(inputs) < n_dev + n_test + 61:
        raise ValueError('Insufficient distinct inputs for construction, development and test')
    order = np.random.default_rng(seed).permutation(len(inputs))
    sections = {'test': order[:n_test], 'dev': order[n_test:n_test+n_dev],
                'construction': order[n_test+n_dev:]}
    return {name: [p for i in indices for p in groups[inputs[i]]] for name, indices in sections.items()}


def select_layer(predictions, targets):
    """Only development predictions belong here; low layer wins ties."""
    if not targets:
        raise ValueError('Empty development set')
    for preds in predictions.values():
        if len(preds) != len(targets):
            raise ValueError('Prediction/target length mismatch')
    return min(predictions, key=lambda layer: (-sum(p == t for p, t in zip(predictions[layer], targets)), int(layer)))


def score(predictions, targets):
    if not targets or len(predictions) != len(targets):
        raise ValueError('Prediction/target length mismatch or empty targets')
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)
