import torch
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ---------------------------------------------
STATE_PT = Path("/home/elefort/real2sim/rsrd/outputs/cooler_5/state.pt")
CKPT_PT  = Path("/home/elefort/real2sim/rsrd/outputs/cooler_5/dig/2025-04-03_094939/nerfstudio_models/step-000014999.ckpt")
OUT_PT   = STATE_PT.with_name("state_clr.pt")
# ---------------------------------------------------------------------

def extract_gauss_from_ckpt(ckpt_dict):
    """Return a dict with gauss_params tensors from the flattened ckpt keys."""
    pipe = ckpt_dict["pipeline"]
    gauss = {}
    prefix = "_model.gauss_params."
    for k, v in pipe.items():
        if k.startswith(prefix):
            sub = k[len(prefix):]  # e.g. 'features_dc'
            gauss[sub] = v
    assert "means" in gauss, "Could not locate gauss_params.means in ckpt"
    return gauss


def build_lut(means_tensor):
    """Build LUT mapping exact (x,y,z) tuples → list[index]."""
    lut = defaultdict(list)
    means_np = means_tensor.detach().cpu().numpy()
    for idx, triplet in enumerate(means_np):
        lut[tuple(triplet)].append(idx)
    return lut


def main():
    state_small = torch.load(STATE_PT, map_location="cpu", weights_only=False)
    ckpt = torch.load(CKPT_PT, map_location="cpu", weights_only=False)
    gauss_full = extract_gauss_from_ckpt(ckpt)

    means_full  = gauss_full["means"]
    means_small = state_small["means"]

    lut = build_lut(means_full)

    mapping = []
    for m in means_small:
        key = tuple(float(v) for v in m.tolist())
        mapping.append(lut[key].pop() if key in lut and lut[key] else -1)

    missing = sum(1 for x in mapping if x == -1)
    print(f"Matched {len(mapping)-missing} / {len(mapping)} gaussians by exact XYZ equality.")
    assert missing == 0, "Some gaussians not matched; exact equality failed." 

    idx = torch.tensor(mapping, dtype=torch.long)

    new_state = {k: v.clone() for k, v in state_small.items()}
    for fld in ("features_dc", "features_rest"):
        if fld in gauss_full:
            new_state[fld] = gauss_full[fld][idx]
            print(f"Replaced {fld} from ckpt.")

    torch.save(new_state, OUT_PT)
    print(f"Wrote recoloured state to {OUT_PT}")

if __name__ == "__main__":
    main()
