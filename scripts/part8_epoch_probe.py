"""Part 8 - price the 40-epoch cap: one moe_full seed at 120 epochs, everything else held fixed.

Every graph model stopped at the cap while the MLP converged well inside it, so "this architecture
underperforms" was tangled up with "this architecture was undertrained". One knob changes - max_epochs -
because a probe that moves two cannot attribute its result to either.
"""
import json
import sys

from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.train.loop import TrainConfig, train_one

MAX_EPOCHS = 120
SEED = 0
TAG = "_probe120"
KEY = "normalised_r2_observed_active"


def main() -> int:
    ensure_free_space()
    lr = json.loads((PROCESSED / "p2_lr.json").read_text())["moe"]
    policy = json.loads((PROCESSED / "p2_loss_mask.json").read_text())["policy"]
    # patience stays at its default 8: the probe must differ from the headline in max_epochs alone, and if
    # early stopping fires here that is itself the answer - the headline runs simply had not converged
    cfg = TrainConfig(model="moe_full", loss_mask=policy, lr=lr, seed=SEED, max_epochs=MAX_EPOCHS, tag=TAG)
    print(f"probe: moe_full seed {SEED}, lr {lr}, loss_mask {policy}, max_epochs {MAX_EPOCHS}", flush=True)
    m = train_one(cfg)
    print(f"{KEY}={m[KEY]} best_epoch={m['history']['best_epoch']} of {MAX_EPOCHS} dir={m['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
