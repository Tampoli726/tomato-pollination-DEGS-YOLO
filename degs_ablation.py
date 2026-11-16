"""Run the eight DEGS-YOLO ablations from the compare folder."""

import argparse
from copy import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from statistics import mean, stdev
import subprocess
import sys
from tempfile import TemporaryDirectory


GROUPS = {
    "A0": "YOLOv8n",
    "A1": "DDS",
    "A2": "DDS + depth control",
    "A3": "DDS + EMA",
    "A4": "DDS + EMA + Slim-neck",
    "A5": "DDS + EMA + Slim-neck + GhostConv",
    "A6": "DEGS-YOLO, w-DIoU alpha=1",
    "A7": "DDS + EMA + Slim-neck + w-DIoU, no GhostConv",
}
HERE = Path(__file__).resolve().parent
TRAINING = dict(
    imgsz=640, batch=4, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, workers=0, amp=False, deterministic=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
    scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=10,
)


def make_config(group, complete, package):
    import yaml

    source = package / "cfg/models/v8/yolov8.yaml" if group == "A0" else complete
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update(nc=1, scale="n")
    if group == "A0":
        return config
    layers = config["backbone"] + config["head"]
    if group not in {"A5", "A6"}:
        for index in (5, 7):
            layers[index][2] = "Conv"
    if group in {"A1", "A2", "A3"}:
        for index in (12, 18):
            layers[index][2] = "C2f"
        layers[16][2] = "Conv"
    if group in {"A1", "A2"}:
        layers[15][2] = "C2f"
    if group == "A1":
        layers[15][1] = 3
    return config


def worker(args):
    sys.path.insert(0, str(args.source))
    import degs_rebuild as rebuild
    import yaml

    with TemporaryDirectory(prefix="degs-ablation-") as directory:
        workspace = Path(directory)
        complete = rebuild.prepare_runtime(workspace)
        package = workspace / "ultralytics"
        model_config = make_config(args.group, complete, package)
        model_file = workspace / f"{args.group}-yolov8n.yaml"
        model_file.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
        sys.path.insert(0, str(workspace))
        config_home = workspace / "settings"
        config_home.mkdir()
        if not args.check:
            config_home = args.run_dir / "settings"
            config_home.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_home)
        # Each run has its own config directory. Seed it with a local Windows
        # font so dataset checks and plotting do not require a font download.
        local_font = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
        if local_font.is_file():
            font_dir = config_home / "Ultralytics"
            font_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_font, font_dir / "Arial.ttf")
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG
        from ultralytics.utils.torch_utils import get_flops, init_seeds
        import torch

        torch.set_num_threads(2)
        init_seeds(args.seed, deterministic=True)
        model = YOLO(str(model_file))
        alpha = 1.0 if args.group in {"A6", "A7"} else None
        if args.check:
            network = model.model
            assert network.stride.tolist() == ([8, 16, 32] if args.group == "A0" else [8, 16])
            if args.group != "A0":
                assert len(network.model[15].m) == (1 if args.group == "A1" else 2)
            network.args = copy(DEFAULT_CFG)
            network.args.w_diou_alpha = alpha
            network.train()
            batch = dict(img=torch.rand(2, 3, 96, 96), batch_idx=torch.tensor([0, 1]),
                         cls=torch.zeros(2, 1), bboxes=torch.tensor([[.5, .5, .3, .3], [.5, .5, .3, .3]]))
            losses, _ = network(batch)
            assert torch.isfinite(losses).all()
            losses.sum().backward()
            assert network.criterion.bbox_loss.w_diou_alpha == alpha
            if args.group in {"A3", "A4", "A5", "A6", "A7"}:
                grad = network.model[15].ema.conv3x3.weight.grad
                assert grad is not None and grad.abs().sum() > 0
            else:
                assert not any(module.__class__.__name__ == "EMA" for module in network.modules())
            print(f"PASS {args.group}: {GROUPS[args.group]}, parameters={sum(p.numel() for p in network.parameters())}")
            return

        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / model_file.name).write_text(model_file.read_text(encoding="utf-8"), encoding="utf-8")
        provenance = dict(group=args.group, seed=args.seed, epochs=args.epochs, training=TRAINING,
                          device=args.device, data=str(args.data), weights=str(args.weights),
                          dataset_yaml_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                          weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                          ultralytics="8.4.92", torch=torch.__version__, python=sys.version,
                          w_diou_alpha=alpha)
        provenance["source_sha256"] = {
            str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.source / "degs_rebuild.py", rebuild.ARCHIVE / "nn/modules/block.py",
                      rebuild.ARCHIVE / "utils/metrics.py", rebuild.ARCHIVE / "所有模型/v8/yolov8-DLp3_C2f-EMA_Ghostconv_slimneck.yaml")
        }
        (args.run_dir / "protocol.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        parameters = sum(p.numel() for p in model.model.parameters())
        gflops = float(get_flops(model.model, imgsz=640))
        model.load(str(args.weights))
        model.train(data=str(args.data), epochs=args.epochs, seed=args.seed, device=args.device,
                    patience=args.epochs, w_diou_alpha=alpha, **TRAINING,
                    project=str(args.run_dir), name="train", exist_ok=False)
        best = Path(model.trainer.save_dir) / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training finished without best.pt.")
        trained = YOLO(str(best))
        metrics = trained.val(data=str(args.data), split="test", imgsz=640, batch=4,
                              device=args.device, workers=0, save_json=True, plots=True,
                              project=str(args.run_dir), name="test", exist_ok=False,
                              conf=0.001, iou=0.7, max_det=300, half=False)
        result = dict(group=args.group, seed=args.seed, epochs=args.epochs, checkpoint=str(best),
                      precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                      parameters=parameters, gflops=gflops,
                      inference_ms=float(metrics.speed["inference"]))
        (args.run_dir / "test_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def summarize(root):
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/test_result.json"))]
    if not results:
        return
    with (root / "all_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    rows = []
    for group in GROUPS:
        runs = [r for r in results if r["group"] == group]
        if not runs:
            continue
        row = dict(group=group, n=len(runs))
        for metric in ("precision", "recall", "map50", "map50_95"):
            values = [r[metric] for r in runs]
            row[metric + "_mean"] = mean(values)
            row[metric + "_sd"] = stdev(values) if len(values) > 1 else ""
        rows.append(row)
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "pollination tamato")
    parser.add_argument("--data", type=Path, default=HERE / "dataset.yaml")
    parser.add_argument("--weights", type=Path, default=HERE / "yolov8n.pt")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Required for training: one or more random seeds, e.g. --seeds 0 1 2 3.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tag", default="ablation_100epochs")
    parser.add_argument("--output", type=Path, default=HERE / "runs_degs_ablation")
    parser.add_argument("--check", action="store_true", help="Check forward/backward only, without training.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--group", choices=GROUPS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.check and not args.worker and args.seeds is None:
        parser.error("Training requires --seeds, e.g. --seeds 0 1 2 3.")
    for key in ("source", "data", "weights", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if not (args.source / "degs_rebuild.py").is_file() or not (args.source / "All-code").is_dir():
        parser.error("--source must contain degs_rebuild.py and All-code.")
    if args.epochs < 1 or (args.seeds is not None and
                           (min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds))):
        parser.error("Use positive epochs and unique nonnegative seeds.")
    if len(set(args.groups)) != len(args.groups):
        parser.error("Groups must be unique.")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("Use a single GPU index or cpu.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag accepts letters, digits, underscores and hyphens.")
    if not args.check and not (args.data.is_file() and args.weights.is_file()):
        parser.error("Dataset YAML and pretrained weights must exist.")
    if args.worker:
        worker(args)
        return
    root = args.output / args.tag
    if not args.check:
        root.mkdir(parents=True, exist_ok=False)
    try:
        for seed in ([args.seeds[0] if args.seeds else 0] if args.check else args.seeds):
            for group in args.groups:
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--group", group,
                           "--source", str(args.source), "--data", str(args.data), "--weights", str(args.weights),
                           "--seed", str(seed), "--epochs", str(args.epochs), "--device", args.device,
                           "--run-dir", str(root / f"{group}_seed{seed}")]
                if args.check:
                    command.append("--check")
                subprocess.run(command, check=True)
    finally:
        if not args.check:
            summarize(root)


if __name__ == "__main__":
    main()
