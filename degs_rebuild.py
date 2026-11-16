"""Rebuild and train DEGS-YOLO from the archived All-code snapshot."""

import argparse
import ast
from copy import copy
from importlib.metadata import version
import importlib.util
import math
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "All-code"
BASE_VERSION = "8.4.92"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected source layout near {old!r}")
    return source.replace(old, new, 1)


def archived_definitions(path, names):
    source = path.read_text(encoding="utf-8-sig")
    definitions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    return "\n\n".join(definitions[name] for name in names)


def prepare_runtime(workspace):
    spec = importlib.util.find_spec("ultralytics")
    if not spec or not spec.origin or version("ultralytics") != BASE_VERSION:
        raise SystemExit(f"Install Ultralytics {BASE_VERSION} in the training environment.")

    package = workspace / "ultralytics"
    shutil.copytree(Path(spec.origin).parent, package, ignore=shutil.ignore_patterns("__pycache__"))

    modules = archived_definitions(
        ARCHIVE / "nn/modules/block.py",
        ("Conv_Mish", "GSConv", "GSBottleneck", "VoVGSCSP", "EMA", "C2f_EMA"),
    )
    (package / "nn/modules/degs.py").write_text(
        "import torch\nfrom torch import nn\nfrom .block import Bottleneck\n"
        "from .conv import Conv, autopad\n\n" + modules + "\n",
        encoding="utf-8",
    )

    tasks = package / "nn/tasks.py"
    source = tasks.read_text(encoding="utf-8")
    source = replace_once(
        source, "from ultralytics.nn.modules import (",
        "from ultralytics.nn.modules.degs import C2f_EMA, GSConv, VoVGSCSP\n"
        "from ultralytics.nn.modules import (",
    )
    base_modules = "    base_modules = frozenset(\n        {"
    repeat_modules = "    repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {"
    source = replace_once(source, base_modules, base_modules + "\n            C2f_EMA, GSConv, VoVGSCSP,")
    source = replace_once(source, repeat_modules, repeat_modules + "\n            C2f_EMA, VoVGSCSP,")
    tasks.write_text(source, encoding="utf-8")

    loss = package / "utils/loss.py"
    source = loss.read_text(encoding="utf-8")
    source = replace_once(
        source, "def __init__(self, reg_max: int = 16):",
        "def __init__(self, reg_max: int = 16, w_diou_alpha: float | None = None):",
    )
    dfl_init = "        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None\n"
    source = replace_once(source, dfl_init, dfl_init + "        self.w_diou_alpha = w_diou_alpha\n")
    source = replace_once(
        source,
        "        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)",
        "        pred, target = pred_bboxes[fg_mask], target_bboxes[fg_mask]\n"
        "        if self.w_diou_alpha is None:\n"
        "            iou = bbox_iou(pred, target, xywh=False, CIoU=True)\n"
        "        else:\n"
        "            iou = archived_bbox_iou(pred, target, xywh=False, w_diou=True, w_diou_alpha=self.w_diou_alpha)",
    )
    source = replace_once(
        source, "self.bbox_loss = BboxLoss(m.reg_max).to(device)",
        'self.bbox_loss = BboxLoss(m.reg_max, getattr(self.hyp, "w_diou_alpha", None)).to(device)',
    )
    archived_iou = archived_definitions(ARCHIVE / "utils/metrics.py", ("bbox_iou",))
    source += "\n\n" + replace_once(archived_iou, "def bbox_iou(", "def archived_bbox_iou(") + "\n"
    loss.write_text(source, encoding="utf-8")

    defaults = package / "cfg/default.yaml"
    with defaults.open("a", encoding="utf-8") as output:
        output.write("\nw_diou_alpha: null # null selects CIoU, a number selects weighted DIoU\n")

    original = ARCHIVE / "所有模型/v8/yolov8-DLp3_C2f-EMA_Ghostconv_slimneck.yaml"
    model = workspace / "degs-yolov8n.yaml"
    model.write_text(
        replace_once(
            original.read_text(encoding="utf-8-sig"),
            "[-1, 3, C2f_EMA, [256]]", "[-1, 6, C2f_EMA, [256]]",
        ),
        encoding="utf-8",
    )
    return model


def check_model(model):
    import torch
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import BboxLoss

    torch.set_num_threads(2)
    network = model.model
    assert network.stride.tolist() == [8, 16]
    ema = network.model[15].ema
    network.train()
    batch = {
        "img": torch.rand(2, 3, 96, 96),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.35], [0.5, 0.5, 0.35, 0.35]]),
    }
    network.args = copy(DEFAULT_CFG)
    network.args.w_diou_alpha = 1.0
    losses, _ = network(batch)
    assert torch.isfinite(losses).all()
    losses.sum().backward()
    assert ema.conv3x3.weight.grad is not None
    assert ema.conv3x3.weight.grad.abs().sum() > 0
    criterion = network.criterion.bbox_loss
    assert criterion.w_diou_alpha == 1.0

    pred = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
    target = torch.tensor([[[1.0, 0.0, 3.0, 2.0]]])
    aux = (torch.zeros(1, 1, 64), pred, torch.ones(1, 2), target,
           torch.ones(1, 1, 1), torch.tensor(1.0), torch.ones(1, 1, dtype=torch.bool),
           torch.tensor([96.0, 96.0]), torch.ones(1, 1))
    weighted_box, _ = criterion(*aux)
    ciou_box, _ = BboxLoss(16)(*aux)
    torch.testing.assert_close(weighted_box, torch.tensor(1 - 1 / 3 + 2 / 39))
    torch.testing.assert_close(ciou_box, torch.tensor(1 - 1 / 3 + 1 / 13))
    print("PASS: P3/P4 strides, EMA gradients, and distinct w-DIoU/CIoU box losses.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Run a synthetic forward/backward check.")
    mode.add_argument("--evaluate", type=Path, help="Evaluate a rebuilt checkpoint on the test split.")
    parser.add_argument("--data", type=Path, help="Verified YOLO dataset YAML.")
    parser.add_argument("--weights", type=Path, help="Local YOLOv8n pretrained weights.")
    parser.add_argument("--project", type=Path, default=ROOT / "degs_runs", help="Output directory.")
    parser.add_argument("--device", default="0", help="A single GPU index or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--w-diou-alpha", type=float, default=1.0)
    parser.add_argument("--ciou", action="store_true", help="Use CIoU for the loss ablation.")
    args = parser.parse_args()
    if not args.check and not (args.data and args.data.is_file()):
        parser.error("--data must point to a local YAML file.")
    if args.evaluate and not args.evaluate.is_file():
        parser.error("--evaluate must point to a local rebuilt checkpoint.")
    if args.weights and not args.weights.is_file():
        parser.error("--weights must point to a local checkpoint.")
    if not math.isfinite(args.w_diou_alpha) or args.w_diou_alpha < 0:
        parser.error("--w-diou-alpha must be finite and nonnegative.")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("--device must be a single GPU index or cpu.")

    with TemporaryDirectory(prefix="degs-") as directory:
        workspace = Path(directory)
        model_yaml = prepare_runtime(workspace)
        sys.path.insert(0, str(workspace))
        config_dir = workspace / "config" if args.check else args.project.resolve() / ".ultralytics-config"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import init_seeds

        if args.evaluate:
            YOLO(str(args.evaluate.resolve())).val(
                data=str(args.data.resolve()), split="test", device=args.device,
                imgsz=640, batch=4, workers=0, save_json=True,
                project=str(args.project.resolve()), name=f"test_seed{args.seed}",
            )
            return

        init_seeds(args.seed, deterministic=True)
        model = YOLO(str(model_yaml))
        if args.check:
            check_model(model)
            return

        if args.weights:
            model.load(str(args.weights.resolve()))
        loss_name = "ciou" if args.ciou else "wdiou"
        model.train(
            data=str(args.data.resolve()), device=args.device, seed=args.seed,
            epochs=args.epochs, imgsz=640, batch=4, optimizer="SGD", lr0=0.01,
            workers=0, amp=False, deterministic=True, patience=100,
            w_diou_alpha=None if args.ciou else args.w_diou_alpha,
            project=str(args.project.resolve()), name=f"{loss_name}_seed{args.seed}",
        )
        shutil.copyfile(model_yaml, model.trainer.save_dir / model_yaml.name)


if __name__ == "__main__":
    main()
