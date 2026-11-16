# DEGS-YOLO for greenhouse tomato flower detection

This repository reproduces the **reconstructed** DEGS-YOLO model and its eight ablation groups (A0–A7) for greenhouse tomato flower detection. The original training project was partly lost; this implementation was rebuilt from surviving model definitions and training files. It should not be described as a byte-for-byte copy of the original implementation.

## Repository contents

Place these files in the repository root before running the experiments:

```text
degs_rebuild.py
degs_ablation.py
All-code/nn/modules/block.py
All-code/utils/metrics.py
All-code/所有模型/v8/yolov8-DLp3_C2f-EMA_Ghostconv_slimneck.yaml
```

The comparison experiments additionally use `batch_train_test.py`, `batch_train_test_dds.py`, `dds_yolo11n.yaml`, and `dds_yolo12n.yaml` in the same directory. Do not upload the complete historical `All-code` archive or training output folders merely to satisfy this layout.

## Environment

- Python 3.11 is the documented starting environment. Record the actual Python, PyTorch, CUDA, and GPU versions with each published result.
- Install PyTorch and torchvision for your platform using the [official PyTorch installer](https://pytorch.org/get-started/locally/). A CUDA-enabled installation is required for GPU training.
- Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
```

The reconstruction code checks that `ultralytics==8.4.92` is installed. It builds a temporary, modified copy of that package for training; it does not edit the installed package in place.

## Dataset and pretrained weights

Use a YOLO detection dataset with fixed, disjoint train/validation/test splits:

```text
dataset/
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Create a local `dataset.yaml` in the repository root and set `path` to the absolute location of your dataset:

```yaml
path: E:/your/path/to/dataset
train: train/images
val: val/images
test: test/images
names:
  0: tomato_flower
```

Keep `dataset.yaml` local because it contains a machine-specific path. Obtain the official YOLOv8n pretrained checkpoint and place it at `yolov8n.pt` in the repository root. The dataset and `.pt` weights are excluded from Git by `.gitignore`. If you publish a dataset or trained weights separately, link to their location and record their version and checksum here.

## Check the reconstructed model

Run from the repository root:

```powershell
python .\degs_ablation.py --source . --check
```

This checks the model's forward/backward pass, EMA gradients, detection strides, and the selected box loss without starting training.

## Run the ablation experiment

```powershell
python .\degs_ablation.py --source . --data .\dataset.yaml --weights .\yolov8n.pt --groups A0 A1 A2 A3 A4 A5 A6 A7 --seeds 0 1 2 3 --epochs 100 --device 0 --tag paper_ablation
```

The script uses image size 640, batch size 4, SGD, and a fixed dataset split. Each model/seed pair is trained independently. The best validation checkpoint is evaluated once on the **test** split. Outputs are written to `runs_degs_ablation/paper_ablation/`, including each run's `protocol.json`, `test_result.json`, and the aggregate `all_runs.csv` and `summary.csv`. Use a new `--tag` for another run; the script does not reuse an existing tag directory. Set `--device cpu` only for a functional check or a machine without CUDA; a full 100-epoch experiment will be slow on CPU.

| Group | Change relative to preceding stages |
| --- | --- |
| A0 | Standard YOLOv8n |
| A1 | DDS detection head |
| A2 | DDS head with changed depth; no EMA |
| A3 | A2 + EMA |
| A4 | A3 + Slim-neck |
| A5 | A4 + GhostConv |
| A6 | A5 + w-DIoU (`alpha=1`); complete DEGS-YOLO |
| A7 | A4 + w-DIoU, without GhostConv |

These labels describe what the supplied script actually constructs. A2 changes the depth of the DDS design, and A7 helps isolate the GhostConv contribution in the presence of w-DIoU.

## Baselines and DDS transfer

`batch_train_test.py` trains official YOLOv8n, YOLO11n, and YOLO12n baselines. `batch_train_test_dds.py` trains DDS-head versions of YOLO11n and YOLO12n. Put their official pretrained `.pt` checkpoints in the repository root before running them. **Check the `SEEDS`, output paths, and other constants in both scripts before publication:** the current baseline script contains a seed-3-only repeat setting. Use the same data split and matching training settings for comparisons. Record results separately for each seed and model.

## Results and reproducibility

Publish one row per model and seed in `results/per_run_metrics.csv`, with at least `group`, `seed`, `precision`, `recall`, `map50`, and `map50_95`. State whether each row came from the independent test set. Include parameter count, GFLOPs, weight size, and measurement hardware when reporting lightweight or speed results. Do not replace the per-run values with only group means.

The code and model configurations should be cited with a permanent GitHub release or commit identifier when used in a paper. The tomato flower images are not included in this code repository.

## License and attribution

The repository uses the GNU Affero General Public License v3.0 text in `LICENSE`. Before publishing the archived files from `All-code`, verify their provenance and retain any original copyright and license notices. The repository license does not remove third-party obligations.
