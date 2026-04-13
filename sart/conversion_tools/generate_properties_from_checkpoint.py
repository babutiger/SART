"""Generate repository property files from correctly classified MNIST samples."""

import argparse
import csv
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union, Iterable, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_PTH_PATH = "../models/VNNCOMP_ERAN/mnist_9_100_nat.pth"
DEFAULT_USE_NORMALIZE = False
DEFAULT_NUM_SAMPLES = 110
DEFAULT_BATCH_SIZE = 64
DEFAULT_PREFER_INPUT_DIM = 784
DEFAULT_PROPERTIES_RESULTS_CSV = "properties_generation_results.csv"
DEFAULT_PROPERTIES_GROUNDTRUTH_TXT = "properties_groundtruth.txt"


def to_tensor(x):
    if isinstance(x, torch.nn.Parameter):
        return x.data
    if isinstance(x, torch.Tensor):
        return x
    # numpy arrays and similar tensor-like inputs
    try:
        import numpy as np  # noqa
        if isinstance(x, np.ndarray):
            return torch.tensor(x)
    except Exception:
        pass
    return None


def looks_like_state_dict(d: Dict[Any, Any]) -> bool:
    if not isinstance(d, dict) and not isinstance(d, OrderedDict):
        return False
    if not d:
        return False
    good = 0
    for k, v in d.items():
        if not isinstance(k, str):
            return False
        tv = to_tensor(v)
        if tv is None:
            return False
        good += 1
    return good > 0


def list_pairs_to_sd(lst: List) -> Union[Dict[str, torch.Tensor], None]:
    """
    Try to rebuild a `state_dict` from a list-like checkpoint structure.

    Supported forms:
      - [(name, tensor), ...]
      - [{'name':..., 'tensor':...}] / {'key':...,'value':...} / {'name':...,'weight':...,'bias':...}
      - [OrderedDict(...)] / [{'a': OrderedDict(...)}]
      - [keys, values] (two-part form: the first item is a list of strings and
        the second item is a list of tensors)
    """
    # Case A: the top-level list contains one nested state_dict or wrapper.
    if len(lst) == 1:
        single = lst[0]
        if isinstance(single, (dict, OrderedDict)):
            if looks_like_state_dict(single):
                return {k: to_tensor(v) for k, v in single.items()}
            # The state_dict may be nested one level deeper.
            for k, v in single.items():
                if isinstance(v, (dict, OrderedDict)) and looks_like_state_dict(v):
                    return {kk: to_tensor(vv) for kk, vv in v.items()}
        elif isinstance(single, (list, tuple)):
            # Keep trying to unwrap nested list/tuple structures.
            cand = list_pairs_to_sd(list(single))
            if cand is not None:
                return cand

    out: Dict[str, torch.Tensor] = {}
    ok = False

    # Case B: [(name, tensor), ...]
    if all(isinstance(it, (list, tuple)) and len(it) == 2 for it in lst):
        for k, v in lst:
            if isinstance(k, str):
                tv = to_tensor(v)
                if tv is not None:
                    out[k] = tv
                    ok = True
        if ok and out:
            return out

    # Case C: a list of dictionaries
    if all(isinstance(it, dict) for it in lst):
        tmp: Dict[str, torch.Tensor] = {}
        for item in lst:
            # One list item may already store a full mapping.
            if looks_like_state_dict(item):
                tmp.update({k: to_tensor(v) for k, v in item.items()})
                ok = True
                continue
            # Common key patterns: name/key/param paired with tensor/value/data.
            name = None
            for kname in ("name", "key", "param"):
                if kname in item and isinstance(item[kname], str):
                    name = item[kname]
                    break
            if name is not None:
                grabbed = False
                for vname in ("tensor", "value", "data", "param", "weight"):
                    if vname in item:
                        tv = to_tensor(item[vname])
                        if tv is not None:
                            tmp[name] = tv
                            grabbed = True
                            ok = True
                            break
                # Handle nested forms such as {'name': 'fc1', 'weight': W, 'bias': b}.
                for subk in ("weight", "bias"):
                    if subk in item:
                        tv = to_tensor(item[subk])
                        if tv is not None:
                            tmp[f"{name}.{subk}"] = tv
                            ok = True
                if grabbed:
                    continue
        if ok and tmp:
            return tmp

    # Case D: [keys, values]
    if len(lst) == 2 and isinstance(lst[0], (list, tuple)) and isinstance(lst[1], (list, tuple)):
        keys, vals = lst
        if all(isinstance(k, str) for k in keys):
            tvs = [to_tensor(v) for v in vals]
            if all(tv is not None for tv in tvs) and len(keys) == len(tvs):
                return {k: v for k, v in zip(keys, tvs)}

    # Case E: nested OrderedDict / dict values inside the list
    collected: Dict[str, torch.Tensor] = {}
    for it in lst:
        if isinstance(it, (dict, OrderedDict)) and looks_like_state_dict(it):
            collected.update({k: to_tensor(v) for k, v in it.items()})
            ok = True
    if ok and collected:
        return collected

    return None


def deep_extract_state_dict(obj: Any, depth: int = 0, seen: Set[int] = None) -> Dict[str, torch.Tensor]:
    """
    Recursively search nested checkpoint structures and rebuild a usable
    `state_dict` when possible.

    Priority order:
      - dicts that already look like a state_dict
      - a dict field named `state_dict`
      - list-like forms handled by `list_pairs_to_sd`
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        raise RuntimeError("Detected a cyclic checkpoint structure.")
    seen.add(oid)

    # The object is already a state_dict.
    if isinstance(obj, (dict, OrderedDict)) and looks_like_state_dict(obj):
        return {k: to_tensor(v) for k, v in obj.items()}

    # For dictionaries, try common wrapper keys first.
    if isinstance(obj, (dict, OrderedDict)):
        # First scan common wrapper keys.
        for key in ("state_dict", "model_state", "model_state_dict", "net", "weights", "params"):
            if key in obj:
                try:
                    return deep_extract_state_dict(obj[key], depth + 1, seen)
                except Exception:
                    pass
        # Otherwise recurse into each value.
        for v in obj.values():
            try:
                return deep_extract_state_dict(v, depth + 1, seen)
            except Exception:
                continue
        raise RuntimeError("Could not find a usable state_dict.")

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        # First try to interpret the object directly as a list-form state_dict.
        sd = list_pairs_to_sd(list(obj))
        if sd is not None and looks_like_state_dict(sd):
            return sd
        # Otherwise recurse into each element.
        for it in obj:
            try:
                return deep_extract_state_dict(it, depth + 1, seen)
            except Exception:
                continue
        raise RuntimeError("Could not find a usable state_dict.")

    # Other object types are unsupported here.
    raise RuntimeError("Could not find a usable state_dict.")


def build_mlp_from_sd(
    sd: Dict[str, torch.Tensor],
    prefer_input_dim: int = DEFAULT_PREFER_INPUT_DIM,
) -> nn.Module:
    """
    Reconstruct an MLP automatically from all 2D Linear weights in a state_dict.

    Layers are chained by matching `(in_features -> out_features)`. The
    preferred chain start is `in_features == prefer_input_dim`; otherwise the
    code falls back to an input size that is not produced by any previous layer.
    """
    # Collect the shape of every Linear-layer candidate.
    nodes: List[Tuple[str, str, int, int]] = []  # (wkey, bkey, in_f, out_f)
    for k, v in sd.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            base = k[:-7]
            bkey = base + ".bias"
            if bkey in sd and isinstance(sd[bkey], torch.Tensor) and sd[bkey].ndim == 1:
                in_f = v.shape[1]
                out_f = v.shape[0]
                nodes.append((k, bkey, in_f, out_f))

    if not nodes:
        raise RuntimeError("No 2D Linear weights were found in the state_dict; the MLP cannot be rebuilt automatically.")

    # Build connectivity information and choose the chain start.
    in_set = {n[2] for n in nodes}
    out_set = {n[3] for n in nodes}
    starters = [n for n in nodes if n[2] not in out_set] or nodes
    starters.sort(key=lambda x: (x[2] != prefer_input_dim, x[2]))

    # Grow the longest reachable chain.
    def grow_chain(start):
        chain = [start]
        cur_out = start[3]
        used = {(start[0], start[1])}
        while True:
            nxts = [n for n in nodes if n[2] == cur_out and (n[0], n[1]) not in used]
            if not nxts:
                break
            nxt = nxts[0]
            chain.append(nxt)
            used.add((nxt[0], nxt[1]))
            cur_out = nxt[3]
        return chain

    best = []
    for st in starters:
        c = grow_chain(st)
        if len(c) > len(best):
            best = c
    chain = best
    if len(chain) < 1:
        raise RuntimeError("Could not reconstruct a consistent Linear-layer chain from the saved shapes.")

    # Build the MLP structure from the recovered chain.
    dims = [chain[0][2]] + [n[3] for n in chain]  # [in0, out1, out2, ..., outN]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers)

    # Load the recovered weights into the new module.
    linear_idxs = [i for i, m in enumerate(mlp) if isinstance(m, nn.Linear)]
    assert len(linear_idxs) == len(chain)
    with torch.no_grad():
        for idx, (wkey, bkey, in_f, out_f) in enumerate(chain):
            lin = mlp[linear_idxs[idx]]
            W = sd[wkey]
            b = sd[bkey]
            if lin.weight.shape != W.shape or lin.bias.shape != b.shape:
                raise RuntimeError(
                    f"Shape mismatch: layer {idx} expects {tuple(lin.weight.shape)} but received {tuple(W.shape)}."
                )
            lin.weight.copy_(W)
            lin.bias.copy_(b)

    # Wrap the model so image-shaped inputs are flattened automatically.
    class FlattenMLP(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
        def forward(self, x):
            if x.ndim == 4:
                x = x.view(x.size(0), -1)
            return self.core(x)
    return FlattenMLP(mlp)


def load_model_auto(
    path: str,
    device: torch.device,
    prefer_input_dim: int = DEFAULT_PREFER_INPUT_DIM,
) -> nn.Module:
    # Try TorchScript first.
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        print("[Loader] Loaded as TorchScript")
        return m.to(device)
    except Exception:
        pass

    # Fall back to regular torch.load.
    obj = torch.load(path, map_location=device)

    # If the checkpoint already stores a full nn.Module, return it directly.
    if isinstance(obj, nn.Module):
        print("[Loader] Loaded full nn.Module")
        return obj.to(device).eval()

    # Recursively extract or rebuild a usable state_dict, even from list-like checkpoints.
    sd = deep_extract_state_dict(obj)
    # Rebuild and load the MLP.
    model = build_mlp_from_sd(sd, prefer_input_dim=prefer_input_dim).to(device).eval()
    print("[Loader] Built MLP from weight shapes and loaded parameters.")
    return model


def get_transform(use_normalize: bool = DEFAULT_USE_NORMALIZE):
    ops = [transforms.ToTensor()]
    if use_normalize:
        ops.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(ops)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate repository property files from correctly classified MNIST samples.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_PTH_PATH,
        help="Path to the checkpoint used for sample selection.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory used to store generated property files and metadata.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of correctly classified samples to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used during dataset scanning.",
    )
    parser.add_argument(
        "--prefer-input-dim",
        type=int,
        default=DEFAULT_PREFER_INPUT_DIM,
        help="Preferred input width when reconstructing an MLP from a state_dict.",
    )
    parser.add_argument(
        "--use-normalize",
        action="store_true",
        help="Apply MNIST Normalize((0.1307,), (0.3081,)) to the scanned inputs.",
    )
    parser.add_argument(
        "--results-csv-name",
        default=DEFAULT_PROPERTIES_RESULTS_CSV,
        help="Filename for the per-sample metadata CSV inside the save directory.",
    )
    parser.add_argument(
        "--groundtruth-name",
        default=DEFAULT_PROPERTIES_GROUNDTRUTH_TXT,
        help="Filename for the ground-truth label text file inside the save directory.",
    )
    return parser.parse_args()


"""
Generate property files from correctly classified samples while also recording
per-sample metadata.

The script keeps:
- the original dataset index
- the true label
- the predicted label
- whether the sample was classified correctly
- the saved property-file index, when the sample is exported

It also saves:
- `properties_generation_results.csv`
- `properties_groundtruth.txt`
"""

def generate_properties(
    model_path,
    save_dir,
    max_samples=100,
    batch_size=DEFAULT_BATCH_SIZE,
    use_normalize=DEFAULT_USE_NORMALIZE,
    prefer_input_dim=DEFAULT_PREFER_INPUT_DIM,
    results_csv_name=DEFAULT_PROPERTIES_RESULTS_CSV,
    groundtruth_name=DEFAULT_PROPERTIES_GROUNDTRUTH_TXT,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    print(f"[Info] Loading: {model_path}")
    print(f"[Info] Saving properties to: {save_dir}")

    # Ensure the output directory exists.
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Reuse the same checkpoint-loading logic as above.
    model = load_model_auto(
        model_path,
        device,
        prefer_input_dim=prefer_input_dim,
    ).eval()

    # Reuse the same test dataset and loader configuration as above.
    testset = datasets.MNIST(
        root="./datasets",
        train=False,
        download=True,
        transform=get_transform(use_normalize=use_normalize),
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    saved_count = 0
    processed_count = 0
    results = []
    gt_labels = []

    # Scan the test set until enough correctly classified samples are collected.
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            _, predicted = torch.max(logits.data, 1)

            for i in range(images.size(0)):
                original_idx = batch_idx * batch_size + i
                gt = labels[i].item()
                pd = predicted[i].item()
                ok = (gt == pd)

                # Record the result for every processed sample.
                results.append((original_idx, gt, pd, ok, saved_count if ok and saved_count < max_samples else -1))
                gt_labels.append(gt)

                # Print per-sample progress.
                print(
                    f"[{original_idx:05d}] GT={gt} pred={pd} {'✓' if ok else '✗'} saved_idx={saved_count if ok and saved_count < max_samples else 'N/A'}")

                # Save a property file only if the sample is correct and more samples are still needed.
                if ok and saved_count < max_samples:
                    # Save the correctly classified sample.
                    file_path = os.path.join(save_dir, f"mnist_property_{saved_count}.txt")
                    with open(file_path, "w") as img_file:
                        # Flatten the image and write the 784 pixel values.
                        flat_image = images[i].view(-1)
                        for j in range(784):
                            img_file.write("%.8f\n" % flat_image[j].item())

                        # Write the 9 adversarial target constraints.
                        true_label = gt
                        for k in range(10):
                            if k == true_label:
                                continue
                            property_list = [0 for _ in range(11)]
                            property_list[true_label] = -1
                            property_list[k] = 1
                            img_file.write(str(property_list)[1:-1].replace(',', ''))
                            img_file.write('\n')

                    saved_count += 1

                processed_count += 1

                # Stop early once enough samples have been collected.
                if saved_count >= max_samples:
                    break

            # Stop the outer loop as well once enough samples have been collected.
            if saved_count >= max_samples:
                break

    # Save the detailed per-sample report.
    with open(os.path.join(save_dir, results_csv_name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_index", "true_label", "pred_label", "correct", "saved_index"])
        for original_idx, t, p, ok, saved_idx in results:
            w.writerow([original_idx, t, p, int(ok), saved_idx])

    # Save the ground-truth labels.
    with open(os.path.join(save_dir, groundtruth_name), "w", encoding="utf-8") as f:
        for idx, gt in enumerate(gt_labels):
            f.write(f"{idx},{gt}\n")

    # Print the summary metrics.
    total_correct = sum(1 for _, _, _, ok, _ in results if ok)
    acc = total_correct / processed_count if processed_count else 0.0
    print(f"\n[Summary] Total={processed_count} Correct={total_correct} Acc={acc * 100:.2f}%")
    print(f"[Summary] Saved {saved_count} correctly classified samples")

    # Print the list of saved samples.
    print("\n[Saved samples]")
    saved_samples = [(orig_idx, gt, pd, saved_idx) for orig_idx, gt, pd, ok, saved_idx in results if saved_idx != -1]
    for orig_idx, gt, pd, saved_idx in saved_samples:
        print(f"Saved index {saved_idx:03d}: Original index {orig_idx:05d}, GT={gt}, pred={pd}")

    return saved_count, results

if __name__ == "__main__":
    args = parse_args()
    generate_properties(
        args.checkpoint,
        args.save_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        use_normalize=args.use_normalize,
        prefer_input_dim=args.prefer_input_dim,
        results_csv_name=args.results_csv_name,
        groundtruth_name=args.groundtruth_name,
    )
