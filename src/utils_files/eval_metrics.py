# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module evaluates the forecasted trajectories against the ground truth."""

import math
from pprint import pprint
from typing import Dict, List, Optional

import numpy as np

from utils_files import utils

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.
    The Brier Score is defined here:
        Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly weather review, 1950.
        https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR, brier-minADE, brier-minFDE
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade, brier_min_ade = [], [], []
    min_fde, prob_min_fde, brier_min_fde = [], [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        for j in range(len(pruned_trajectories)):
            ade = get_ade(pruned_trajectories[j][:horizon], v[:horizon])
            if ade < curr_min_ade:
                curr_min_ade = ade
        # curr_min_ade = get_ade(pruned_trajectories[min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade
            )
            brier_min_ade.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade)
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde
            )
            brier_min_fde.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde)

    metric_results["minADE"] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)
    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
        metric_results["brier-minADE"] = sum(brier_min_ade) / len(brier_min_ade)
        metric_results["brier-minFDE"] = sum(brier_min_fde) / len(brier_min_fde)
    return metric_results


def get_drivable_area_compliance(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    city_names: Dict[int, str],
    max_n_guesses: int,
) -> float:
    """Compute drivable area compliance metric.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        city_names: Dict mapping sequence id to city name.
        max_n_guesses: Maximum number of guesses allowed.

    Returns:
        Mean drivable area compliance

    """
    avm = ArgoverseMap()

    dac_score = []

    for seq_id, trajectories in forecasted_trajectories.items():
        city_name = city_names[seq_id]
        num_dac_trajectories = 0
        n_guesses = min(max_n_guesses, len(trajectories))
        for trajectory in trajectories[:n_guesses]:
            raster_layer = avm.get_raster_layer_points_boolean(trajectory, city_name, "driveable_area")
            if np.sum(raster_layer) == raster_layer.shape[0]:
                num_dac_trajectories += 1

        dac_score.append(num_dac_trajectories / n_guesses)

    return sum(dac_score) / len(dac_score)


def compute_forecasting_metrics(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    city_names: Dict[int, str],
    max_n_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute all the forecasting metrics.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        city_names: Dict mapping sequence id to city name.
        max_n_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Miss threshold
        forecasted_probabilities: Normalized Probabilities associated with each of the forecasted trajectories.

     Returns:
        metric_results: Dictionary containing values for all metrics.
    """
    metric_results = get_displacement_errors_and_miss_rate(
        forecasted_trajectories,
        gt_trajectories,
        max_n_guesses,
        horizon,
        miss_threshold,
        forecasted_probabilities,
    )
    metric_results["DAC"] = get_drivable_area_compliance(forecasted_trajectories, city_names, max_n_guesses)

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")
    print("------------------------------------------------")
    print(metric_results)
    print("------------------------------------------------")

    return metric_results


def post_eval(args, file2pred, file2labels, file2pred_score, DEs):
    score_file = args.model_recover_path.split('/')[-1]
    if "nuscenes" in args.other_params:
        from utils_files import eval_metrics
        metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, args.mode_num, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
        print(f'metric_results (mode_num = {args.mode_num}):')
        pprint(metric_results)
        if args.mode_num == 5:
            extra_mode_num = 10
            metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, extra_mode_num, args.future_frame_num, 2.0, file2pred_score)
            utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
            print(f'metric_results (mode_num = {extra_mode_num}):')
            pprint(metric_results)
        if args.mode_num == 10:
            extra_mode_num = 5
            metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, extra_mode_num, args.future_frame_num, 2.0, file2pred_score)
            utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
            print(f'metric_results (mode_num = {extra_mode_num}):')
            pprint(metric_results)
        extra_mode_num = 1
        metric_results = eval_metrics.get_displacement_errors_and_miss_rate(file2pred, file2labels, extra_mode_num, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
        print(f'metric_results (mode_num = {extra_mode_num}):')
        pprint(metric_results)
    else:
        from argoverse.evaluation import eval_forecasting
        metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, args.mode_num, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
        print("brier-minFDE",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE",('%.4f' % metric_results["minADE"]),",minFDE",('%.4f' % metric_results["minFDE"]),",MR",('%.4f' % metric_results["MR"]))
        metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 1, args.future_frame_num, 2.0, file2pred_score)
        utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
        print("brier-minFDE1",('%.4f' % metric_results["brier-minFDE"]), ",brier-minADE1",('%.4f' % metric_results["brier-minADE"]),\
            ",minADE1",('%.4f' % metric_results["minADE"]),",minFDE1",('%.4f' % metric_results["minFDE"]),",MR1",('%.4f' % metric_results["MR"]))
