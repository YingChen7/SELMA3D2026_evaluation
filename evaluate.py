"""
Example evaluation method for Grand Challenge.

This is the file you should edit to implement your evaluation logic.
It is called by app.py when the /invoke endpoint is hit.

  main() — Called once per evaluation. Read algorithm outputs
           from /input, compute metrics, write results to /output.

The evaluation steps are:

  1. Read algorithm outputs and associate them with ground truths via predictions.json
  2. Calculate metrics by comparing the algorithm output to the ground truth
  3. Repeat for all algorithm jobs that ran for this submission
  4. Aggregate the calculated metrics
  5. Save the metrics to /output/metrics.json

To test locally:  ./do_test_run.sh
To save for upload to Grand Challenge:   ./do_save.sh

Any implementation will do as long as it produces metrics.json in the expected format.

For more details see:
  https://grand-challenge.org/documentation/evaluation/
  https://grand-challenge.org/documentation/runtime-environment/
"""

import glob
import json
import logging
import random
from pathlib import Path
from pprint import pformat
from statistics import mean
import numpy as np
import SimpleITK
from helpers import run_prediction_processing, setup_logger, tree
from panoptica import (
    InputType,
    Panoptica_Evaluator,
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
)
from panoptica.metrics import Metric
from panoptica.metrics.cldice import _compute_centerline_dice_coefficient
from statistics import mean, stdev

logger = logging.getLogger("evaluate")


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")


def main():
    setup_logger(
        # Optionally: change this to the more verbose DEBUG
        level=logging.INFO,
    )

    log_inputs()

    metrics = {}
    predictions = read_predictions()


    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process, predictions=predictions)

    # We already have the results per prediction. Now we can aggregate the results and
    # generate an overall score(s) for this submission
    iso_dice_scores, iso_pq_scores, con_dice_scores, con_cldice_scores = [], [], [], []
        
    for result in metrics["results"]:
        if "panoptic_quality" in result:
            iso_dice_scores.append(result["dice"])
            iso_pq_scores.append(result["panoptic_quality"])
        elif "cldice" in result:
            con_dice_scores.append(result["dice"])
            con_cldice_scores.append(result["cldice"])
        else:
            raise RuntimeError("Unknown category during aggregation")
    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    metrics["aggregates"] = {
        "iso_dice_mean":              mean(iso_dice_scores),
        "iso_dice_std":               stdev(iso_dice_scores),
        "iso_panoptic_quality_mean":  mean(iso_pq_scores),
        "iso_panoptic_quality_std":   stdev(iso_pq_scores),
        "con_dice_mean":              mean(con_dice_scores),
        "con_dice_std":               stdev(con_dice_scores),
        "con_cldice_mean":            mean(con_cldice_scores),
        "con_cldice_std":             stdev(con_cldice_scores),
    }
    

    # Make sure to save the metrics
    # `metrics` must be a dictionary (or any object serializable to JSON via json.dumps)
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key(job)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("light-sheet-3d-microscopy",): process_interf0,
    }[interface_key]

    # Call the handler
    return handler(job)


def process_interf0(
    job,
):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results

    location_isolated_biological_structure = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="isolated-biological-structure",
    )

    # Secondly, read the results

    prediction = load_image_file_as_array(
        location=location_isolated_biological_structure,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth

    image_name_light_sheet_3d_microscopy = get_image_name(
        values=job["inputs"],
        slug="light-sheet-3d-microscopy",
    )

    # Fourthly, load your ground truth

    # Your ground truth will be extracted to the `ground_truth_dir` at runtime on Grand Challenge
    # Note: when testing locally, the local `./ground_truth` directory is mounted here
    # Eventually, you should upload it as a tarball to Grand Challenge!
    # Go to Admin > Phase Settings and upload it under Ground Truths.
    ground_truth_dir = Path("/opt/ml/input/data/ground_truth/task1_gt")
    ground_truth_img = SimpleITK.ReadImage(Path(ground_truth_dir / image_name_light_sheet_3d_microscopy))
    ground_truth = SimpleITK.GetArrayFromImage(ground_truth_img)
    # Convert it to a Numpy array
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch for {image_name_light_sheet_3d_microscopy}: "
            f"prediction={prediction.shape}, GT={ground_truth.shape}"
        )
    prediction[prediction > 0] = 1
    ground_truth[ground_truth > 0] = 1

    # 4. Volumetric Dice is used for every structure type.
    dice = volumetric_dice(prediction, ground_truth)

    if image_name_light_sheet_3d_microscopy.startswith(("axonalmarker", "blood_vessel", "gutnerve", "lymphatic_vessel", "peripheral_nerve")):
        cldice = _compute_centerline_dice_coefficient(ground_truth, prediction)
        result = {"dice": dice,
                    "cldice": cldice}
    elif image_name_light_sheet_3d_microscopy.startswith(("AD_plaques", "cell_nuclei", "cfos_neuron", "microglia", "protein")):
        evaluator = Panoptica_Evaluator(
                expected_input=InputType.SEMANTIC,
                instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=None),
                instance_matcher=NaiveThresholdMatching(matching_threshold=0.3),
                instance_metrics = [Metric.DSC]
            )
        
        evaluator_result = evaluator.evaluate(prediction, ground_truth, verbose=False)["ungrouped"]
        evaluator_result.to_dict()
        result = {"dice": dice,
                    "panoptic_quality":evaluator_result.pq_dsc
                }
    else:
        raise RuntimeError(f"Unknown category: {image_name_light_sheet_3d_microscopy}")

    logger.info(f"Evaluatated {image_name_light_sheet_3d_microscopy}")

    return result

def volumetric_dice(prediction, ground_truth):
    """Binary volumetric Dice similarity coefficient."""
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    pred_sum = int(prediction.sum())
    gt_sum = int(ground_truth.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.logical_and(prediction, ground_truth).sum())
    return float(2.0 * intersection / (pred_sum + gt_sum))

def log_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    # Each interface has a unique key that is the set of socket slugs given as input
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    # `metrics` must be a dict (or any object that is JSON-serializable via json.dumps).
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
