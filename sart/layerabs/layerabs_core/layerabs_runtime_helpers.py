"""Shared runtime helpers for LayerABS propagation modules."""

from __future__ import annotations

import concurrent.futures
import os
import pickle
import time

from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    create_fdl_directory,
)

__all__ = [
    "append_stage_snapshots",
    "finalize_propagation_run",
    "log_stage_execution_time",
    "persist_layerabs_state_snapshot",
    "run_parallel_tasks",
    "run_sorted_parallel_tasks",
]


def run_parallel_tasks(task_runner, task_args, max_workers=None):
    """Execute one propagation stage in parallel and return unsorted results."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(task_runner, task_args))


def run_sorted_parallel_tasks(task_runner, task_args, max_workers=None):
    """Execute one propagation stage in parallel and return index-sorted results."""
    results = run_parallel_tasks(task_runner, task_args, max_workers)
    results.sort(key=lambda item: item[0])
    return results


def log_stage_execution_time(start_time):
    """Print the elapsed runtime for the current propagation stage."""
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time} seconds")


def append_stage_snapshots(snapshot_list, pre_activation_snapshot, post_activation_snapshot=None):
    """Append one or two stage snapshot lists to the cumulative snapshot buffer."""
    snapshot_list.append(pre_activation_snapshot)
    if post_activation_snapshot is not None:
        snapshot_list.append(post_activation_snapshot)


def persist_layerabs_state_snapshot(layerabs_state, filename):
    """Persist one LayerABS state snapshot under the shared `fdl` directory."""
    folder_name = create_fdl_directory()
    temp_file_path = os.path.join(folder_name, filename)
    with open(temp_file_path, "wb") as snapshot_file:
        pickle.dump(layerabs_state, snapshot_file)
    return temp_file_path


def finalize_propagation_run(
    *,
    start_time_all,
    interval_label,
    interval_values,
    layerabs_state,
    snapshot_filename,
    mip_snapshots,
):
    """Print the run summary, persist the state snapshot, and return the common tuple."""
    execution_time_all = time.time() - start_time_all
    print(f"Execution time all: {execution_time_all} seconds")
    print(f"{interval_label}: {interval_values}")
    temp_file_path = persist_layerabs_state_snapshot(
        layerabs_state,
        snapshot_filename,
    )
    return interval_values, temp_file_path, mip_snapshots
