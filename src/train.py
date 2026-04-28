"""Training, evaluation, and result-persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from .metrics import key_rank_curve, traces_to_recover


def fit(model: keras.Model, X, Y, *, epochs: int = 50, batch_size: int = 200,
        validation_split: float = 0.1, verbose: int = 2,
        callbacks: list | None = None) -> keras.callbacks.History:
    return model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, verbose=verbose,
                     callbacks=callbacks or [])


def evaluate_attack(model: keras.Model, X_attack, Y_attack, plaintexts, true_key,
                    *, label_kind: str = "hw", n_attacks: int = 50,
                    max_traces: int = 2000) -> dict:
    """Returns dict: top1, top5, rank_curve (np.ndarray), traces_to_recover."""
    probs = model.predict(X_attack, batch_size=512, verbose=0)
    pred_top1 = probs.argmax(axis=1)
    top1 = float((pred_top1 == Y_attack).mean())
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5 = float((top5_idx == Y_attack[:, None]).any(axis=1).mean())

    rank_curve = key_rank_curve(
        model_probs=probs, plaintexts=plaintexts,
        true_key_byte=int(true_key), label_kind=label_kind,
        n_attacks=n_attacks, max_traces=max_traces,
    )
    return {
        "top1": top1,
        "top5": top5,
        "rank_curve": rank_curve,
        "traces_to_recover": traces_to_recover(rank_curve, threshold=0),
    }


def save_result(out_dir: str | Path, name: str, result: dict, extra: dict | None = None) -> Path:
    """Persist evaluation results: rank curve as .npy, scalars as .json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}_rank.npy", result["rank_curve"])
    payload = {
        "name": name,
        "top1": result["top1"],
        "top5": result["top5"],
        "traces_to_recover": result["traces_to_recover"],
    }
    if extra:
        payload.update(extra)
    with open(out / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)
    return out / f"{name}.json"


def load_result(out_dir: str | Path, name: str) -> dict:
    out = Path(out_dir)
    with open(out / f"{name}.json") as f:
        payload = json.load(f)
    payload["rank_curve"] = np.load(out / f"{name}_rank.npy")
    return payload
