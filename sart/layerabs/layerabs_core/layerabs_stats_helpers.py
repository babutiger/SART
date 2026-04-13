from __future__ import annotations

from sart.layerabs.layerabs_core.layerabs_runtime_helpers import (
    log_stage_execution_time,
    persist_layerabs_state_snapshot,
    run_parallel_tasks,
)


def count_unstable_neurons(four_dimensional_list):
    """Summarize pre-activation unstable neurons layer by layer."""
    per_layer_unstable = []
    per_layer_checked = []
    cumulative_unstable = []
    running_total = 0

    for layer in four_dimensional_list:
        layer_unstable = 0
        layer_checked = 0

        for neuron_slots in layer:
            if not isinstance(neuron_slots, list) or len(neuron_slots) == 0:
                continue

            pre_vec = neuron_slots[0]
            if not isinstance(pre_vec, list) or len(pre_vec) < 5:
                continue

            lb = pre_vec[3]
            ub = pre_vec[4]
            if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
                layer_checked += 1
                if lb < 0 and ub > 0:
                    layer_unstable += 1

        running_total += layer_unstable
        per_layer_unstable.append(layer_unstable)
        per_layer_checked.append(layer_checked)
        cumulative_unstable.append(running_total)

    return {
        "per_layer_unstable": per_layer_unstable,
        "per_layer_checked": per_layer_checked,
        "cumulative_unstable": cumulative_unstable,
    }

def print_unstable_neuron_summary(layerabs_state):
    """Print the per-layer unstable-neuron summary for a LayerABS state."""
    stats = count_unstable_neurons(layerabs_state)

    print("per_layer_unstable     =", stats["per_layer_unstable"])
    print("per_layer_checked      =", stats["per_layer_checked"])
    print("cumulative_unstable    =", stats["cumulative_unstable"])

    for layer_index, (unstable_count, checked_count, cumulative_count) in enumerate(
        zip(
            stats["per_layer_unstable"],
            stats["per_layer_checked"],
            stats["cumulative_unstable"],
        )
    ):
        unstable_ratio = (unstable_count / checked_count) if checked_count else 0.0
        print(
            f"Layer {layer_index}: unstable={unstable_count} / checked={checked_count}  "
            f"(ratio={unstable_ratio:.2%}), cumulative={cumulative_count}"
        )
