from __future__ import annotations

import os
import pickle
import time
from importlib import import_module

from sart.utils.util import save_number_result
from sart.layerabs.support.extract_filename_from_path import (
    extract_filename_from_path,
)
from sart.layerabs.support.timeout import execute_with_timeout
from sart.layerabs.layerabs_core.layerabs_shared_helpers import (
    check_negative_upper_bound,
    replace_tuple_element,
    store_data_by_4dlist,
)


def _print_stage_outcome(property_label, outcome):
    print(f"{property_label} -- StageOutcome: {outcome}")


def run_layerabs_verification(
    *,
    config,
    d,
    style_time,
    network_cls,
    forward_propagation_fn,
    terminate_child_processes,
    average_divisor,
    reported_total,
    print_property_header=False,
    show_running_average=False,
):
    net = network_cls()
    net.load_nnet(config.model_path)

    property_list = config.build_property_list()
    nnet_verify = config.model_path

    os.makedirs("../result/original_result", exist_ok=True)
    result_path = (
        "../result/original_result/"
        + config.result_prefix
        + "_delta_"
        + str(d)
        + "_"
        + str(style_time)
        + ".txt"
    )

    with open(result_path, mode="w+", encoding="utf-8") as file:
        num_ans = 0
        time_ans = 0
        time_max = 0
        processed_properties_count = 0

        maximum_time_threshold = config.maximum_time_threshold

        for property_path in property_list:
            if print_property_header:
                print(f"property_i: {property_path}")

            start_time = time.time()

            # Keep the historical behavior: DeepPoly pre-check is currently bypassed.
            num_single_deeppoly = 0

            if num_single_deeppoly == 1:
                num_single = num_single_deeppoly
                property_label = extract_filename_from_path(property_path)
                print(f"{property_label} -- Verified")
            else:
                arguments = store_data_by_4dlist(nnet_verify, property_path, d)

                print("----##----")
                property_label = extract_filename_from_path(property_path)

                result = execute_with_timeout(
                    maximum_time_threshold,
                    forward_propagation_fn,
                    arguments,
                )

                print("----**----")
                terminate_child_processes()

                if result != 0:
                    num_single = check_negative_upper_bound(result[0])
                else:
                    num_single = 0
                    print(f"{property_label} -- Time out")

                if num_single == 1:
                    print(f"{property_label} -- Verified")
                elif result == 0:
                    print(f"{property_label} -- UnVerified, Time out")
                else:
                    print(f"{property_label} -- UnSafe, Complete")

            num_ans += num_single
            time_single = time.time() - start_time
            print("time:", time_single)
            time_ans += time_single
            if time_single > time_max:
                time_max = time_single

            if show_running_average:
                processed_properties_count += 1
                current_avg_time = time_ans / processed_properties_count
                print(
                    f"Current total time: {time_ans:.4f}s, average time after processing "
                    f"{processed_properties_count}/{reported_total} properties: {current_avg_time:.4f}s"
                )

            save_number_result(property_label, num_single, time_single, file)

        file.write("delta : " + str(d) + "\n")
        file.write("number_sum : " + str(num_ans) + "\n")
        file.write("time_sum : " + str(time_ans) + "\n")
        file.write("time_average : " + str(time_ans / average_divisor) + "\n")
        file.write("time_max : " + str(time_max) + "\n")

    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)
    print("time_average:", time_ans / average_divisor)
    print("time_max:", time_max)


def run_layerabs_refinement_verification(
    *,
    config,
    d,
    style_time,
    refinement_forward_fn,
    complete_forward_fn,
    terminate_child_processes,
):
    network_cls = getattr(import_module(config.network_module), "network")
    net = network_cls()
    net.load_nnet(config.model_path)

    property_list = config.build_property_list()
    reported_total = config.report_total()
    average_divisor = config.resolved_average_divisor()
    nnet_verify = config.model_path

    os.makedirs("../result/original_result", exist_ok=True)
    result_path = (
        "../result/original_result/"
        + config.result_prefix
        + "_delta_"
        + str(d)
        + "_"
        + str(style_time)
        + ".txt"
    )

    with open(result_path, mode="w+", encoding="utf-8") as file:
        num_ans = 0
        time_ans = 0
        time_max = 0
        processed_properties_count = 0

        maximum_time_threshold = config.maximum_time_threshold

        for property_path in property_list:
            start_time = time.time()

            num_single_deeppoly, save_deeppoly = net.find_robustness_number_test(
                property_path, d, TRIM=True
            )
            property_label = extract_filename_from_path(property_path)

            if num_single_deeppoly == 1:
                num_single = num_single_deeppoly
                print(f"{property_label} -- Verified")
                _print_stage_outcome(property_label, "stage1_safe")
            else:
                arguments = store_data_by_4dlist(nnet_verify, property_path, d)
                refinement_arguments = arguments + (save_deeppoly, config.l_mip_num)

                result = execute_with_timeout(
                    maximum_time_threshold,
                    refinement_forward_fn,
                    refinement_arguments,
                )

                print("----##----")
                terminate_child_processes()

                if result != 0:
                    num_single = check_negative_upper_bound(result[0])
                else:
                    num_single = 0
                    print(f"{property_label} -- Time out")

                if num_single == 1:
                    print(f"{property_label} -- Verified")
                    _print_stage_outcome(property_label, "stage2_safe")
                elif result == 0:
                    print(f"{property_label} -- UnVerified, Time out")
                    _print_stage_outcome(property_label, "stage2_timeout")
                else:
                    print(f"{property_label} -- UnVerified, Continue Verify...")

                    save_mip = result[2]
                    file_path = result[1]
                    with open(file_path, "rb") as handle:
                        four_dimensional_list_mip = pickle.load(handle)

                    complete_arguments = replace_tuple_element(
                        refinement_arguments,
                        11,
                        four_dimensional_list_mip,
                    )
                    complete_arguments = replace_tuple_element(
                        complete_arguments,
                        14,
                        save_mip,
                    )

                    previous_time_single = time.time() - start_time
                    remaining_time_threshold = (
                        maximum_time_threshold - previous_time_single
                    )

                    result = execute_with_timeout(
                        remaining_time_threshold,
                        complete_forward_fn,
                        complete_arguments,
                    )

                    print("----**----")
                    terminate_child_processes()

                    if result != 0:
                        num_single = check_negative_upper_bound(result[0])
                    else:
                        num_single = 0
                        print(f"{property_label} -- Time out")

                    if num_single == 1:
                        print(f"{property_label} -- Verified")
                        _print_stage_outcome(property_label, "stage3_safe")
                    elif result == 0:
                        print(f"{property_label} -- UnVerified, Time out")
                        _print_stage_outcome(property_label, "stage3_timeout")
                    else:
                        print(f"{property_label} -- UnSafe, Complete")
                        _print_stage_outcome(property_label, "stage3_unsafe")

            num_ans += num_single
            time_single = time.time() - start_time
            print("time:", time_single)
            time_ans += time_single
            if time_single > time_max:
                time_max = time_single

            if getattr(config, "show_running_average", False):
                processed_properties_count += 1
                current_avg_time = time_ans / processed_properties_count
                print(
                    f"Current total time: {time_ans:.4f}s, average time after processing "
                    f"{processed_properties_count}/{reported_total} properties: {current_avg_time:.4f}s"
                )

            save_number_result(property_label, num_single, time_single, file)

        file.write("delta : " + str(d) + "\n")
        file.write("number_sum : " + str(num_ans) + "\n")
        file.write("time_sum : " + str(time_ans) + "\n")
        file.write("time_average : " + str(time_ans / average_divisor) + "\n")
        file.write("time_max : " + str(time_max) + "\n")

    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)
    print("time_average:", time_ans / average_divisor)
    print("time_max:", time_max)


def run_layerabs_incomplete_verification(
    *,
    config,
    d,
    style_time,
    refinement_forward_fn,
    terminate_child_processes,
    k_layers=2,
):
    network_cls = getattr(import_module(config.network_module), "network")
    net = network_cls()
    net.load_nnet(config.model_path)

    property_list = config.build_property_list()
    reported_total = config.report_total()
    average_divisor = config.resolved_average_divisor()
    nnet_verify = config.model_path
    effective_k_layers = k_layers if k_layers is not None else config.default_k_layers

    os.makedirs("../result/original_result", exist_ok=True)
    result_path = (
        "../result/original_result/"
        + config.result_prefix
        + "_k_"
        + str(effective_k_layers)
        + "_delta_"
        + str(d)
        + "_"
        + str(style_time)
        + ".txt"
    )

    with open(result_path, mode="w+", encoding="utf-8") as file:
        num_ans = 0
        time_ans = 0
        time_max = 0
        processed_properties_count = 0

        maximum_time_threshold = config.maximum_time_threshold

        for property_path in property_list:
            start_time = time.time()

            num_single_deeppoly, save_deeppoly = net.find_robustness_number_test(
                property_path, d, TRIM=True
            )
            property_label = extract_filename_from_path(property_path)

            if num_single_deeppoly == 1:
                num_single = num_single_deeppoly
                print(f"{property_label} -- Verified")
                _print_stage_outcome(property_label, "stage1_safe")
            else:
                arguments = store_data_by_4dlist(nnet_verify, property_path, d)
                refinement_arguments = arguments + (save_deeppoly, effective_k_layers)

                result = execute_with_timeout(
                    maximum_time_threshold,
                    refinement_forward_fn,
                    refinement_arguments,
                )

                print("----##----")
                terminate_child_processes()

                if result != 0:
                    num_single = check_negative_upper_bound(result[0])
                else:
                    num_single = 0
                    print(f"{property_label} -- Time out")

                if num_single == 1:
                    print(f"{property_label} -- Verified")
                    _print_stage_outcome(property_label, "stage2_safe")
                elif result == 0:
                    print(f"{property_label} -- UnVerified, Time out")
                    _print_stage_outcome(property_label, "stage2_timeout")
                else:
                    print(f"{property_label} -- Unknown, Incomplete")
                    _print_stage_outcome(property_label, "stage2_unknown")

            num_ans += num_single
            time_single = time.time() - start_time
            print("time:", time_single)
            time_ans += time_single
            if time_single > time_max:
                time_max = time_single

            if getattr(config, "show_running_average", False):
                processed_properties_count += 1
                current_avg_time = time_ans / processed_properties_count
                print(
                    f"Current total time: {time_ans:.4f}s, average time after processing "
                    f"{processed_properties_count}/{reported_total} properties: {current_avg_time:.4f}s"
                )

            save_number_result(property_label, num_single, time_single, file)

        file.write("delta : " + str(d) + "\n")
        file.write("k_layers : " + str(effective_k_layers) + "\n")
        file.write("number_sum : " + str(num_ans) + "\n")
        file.write("time_sum : " + str(time_ans) + "\n")
        file.write("time_average : " + str(time_ans / average_divisor) + "\n")
        file.write("time_max : " + str(time_max) + "\n")

    print("delta:", d)
    print("k_layers:", effective_k_layers)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)
    print("time_average:", time_ans / average_divisor)
    print("time_max:", time_max)


def run_layerabs_timelimit_verification(
    *,
    config,
    d,
    style_time,
    forward_propagation_fn,
    terminate_child_processes,
):
    network_cls = getattr(import_module(config.network_module), "network")
    net = network_cls()
    net.load_nnet(config.model_path)

    property_list = config.build_property_list()
    reported_total = config.report_total()
    average_divisor = config.resolved_average_divisor()
    nnet_verify = config.model_path

    num_ans = 0
    time_ans = 0
    time_max = 0
    processed_properties_count = 0
    maximum_time_threshold = config.maximum_time_threshold

    for property_path in property_list:
        start_time = time.time()

        num_single_deeppoly, save_deeppoly = net.find_robustness_number_test(
            property_path, d, TRIM=True
        )
        property_label = extract_filename_from_path(property_path)

        if num_single_deeppoly == 1:
            num_single = num_single_deeppoly
            print(f"{property_label} -- Verified")
        else:
            arguments = store_data_by_4dlist(nnet_verify, property_path, d)
            timelimit_arguments = arguments + (save_deeppoly, config.l_mip_num)

            result = execute_with_timeout(
                maximum_time_threshold,
                forward_propagation_fn,
                timelimit_arguments,
            )

            print("----##----")
            terminate_child_processes()

            if result != 0:
                num_single = check_negative_upper_bound(result[0])
            else:
                num_single = 0
                print(f"{property_label} -- Time out")

            if num_single == 1:
                print(f"{property_label} -- Verified")
            else:
                print(f"{property_label} -- UnVerified")

        num_ans += num_single
        time_single = time.time() - start_time
        print("time:", time_single)
        time_ans += time_single
        if time_single > time_max:
            time_max = time_single

        if getattr(config, "show_running_average", False):
            processed_properties_count += 1
            current_avg_time = time_ans / processed_properties_count
            print(
                f"Current total time: {time_ans:.4f}s, average time after processing "
                f"{processed_properties_count}/{reported_total} properties: {current_avg_time:.4f}s"
            )

    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)
    print("time_average:", time_ans / average_divisor)
    print("time_max:", time_max)
