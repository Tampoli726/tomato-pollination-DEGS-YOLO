from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_YAML = BASE_DIR / "dataset.yaml"

# 当前需要比较的官方模型
MODEL_PATHS = [
    BASE_DIR / "yolov8n.pt",
    BASE_DIR / "yolo11n.pt",
    BASE_DIR / "yolo12n.pt",
]

# 训练结果和测试结果的保存位置
TRAIN_ROOT = BASE_DIR / "runs_batch_train"
TEST_ROOT = BASE_DIR / "runs_batch_test"
SUMMARY_ROOT = BASE_DIR / "batch_summary"


# ============================================================
# 2. 统一训练参数
# ============================================================

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
WORKERS = 4
DEVICE = 0

# Seeds and output folders are selected from terminal arguments in main().

# 为保证公平性，所有模型显式使用同一优化器
OPTIMIZER = "SGD"

LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

PATIENCE = 100
AMP = True
DETERMINISTIC = True

# 是否重新训练已经存在best.pt的实验
# False：已有结果时跳过训练，直接测试
# True：允许重新运行并覆盖同名目录中的结果
RETRAIN_EXISTING = False


# ============================================================
# 3. 数据增强参数
# 所有模型必须完全相同
# ============================================================

AUGMENTATION_ARGS: dict[str, Any] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
}


# ============================================================
# 4. 工具函数
# ============================================================

def check_environment() -> None:
    """检查数据、模型和CUDA环境。"""

    print("=" * 70)
    print("项目目录：", BASE_DIR)
    print("数据配置：", DATA_YAML)
    print("PyTorch：", torch.__version__)
    print("Ultralytics：", ultralytics.__version__)
    print("PyTorch CUDA：", torch.version.cuda)
    print("CUDA可用：", torch.cuda.is_available())

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"没有找到数据集配置文件：{DATA_YAML}"
        )

    missing_models = [
        str(path) for path in MODEL_PATHS if not path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "以下模型文件不存在：\n" + "\n".join(missing_models)
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用，不能进行GPU批量训练。"
        )

    if torch.cuda.device_count() <= DEVICE:
        raise RuntimeError(
            f"设备编号{DEVICE}不存在，当前GPU数量为"
            f"{torch.cuda.device_count()}。"
        )

    print("训练GPU：", torch.cuda.get_device_name(DEVICE))
    print(
        "GPU显存：",
        f"{torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3:.2f} GB",
    )
    print("=" * 70)


def save_experiment_config(seeds: list[int]) -> None:
    """保存软硬件环境和统一实验参数。"""

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "gpu_count": torch.cuda.device_count(),
    }

    config = {
        "dataset_yaml": str(DATA_YAML),
        "models": [str(path) for path in MODEL_PATHS],
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "workers": WORKERS,
        "device": DEVICE,
        "seeds": seeds,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "amp": AMP,
        "deterministic": DETERMINISTIC,
        "augmentation": AUGMENTATION_ARGS,
    }

    with open(
        SUMMARY_ROOT / "environment.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)

    with open(
        SUMMARY_ROOT / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def safe_float(value: Any) -> float:
    """将指标安全转换为浮点数。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def calculate_ci95(values: list[float]) -> tuple[float, float]:
    """
    计算95%置信区间。

    小样本使用Student t临界值。
    """

    valid_values = [
        value for value in values if not math.isnan(value)
    ]

    number = len(valid_values)

    if number == 0:
        return float("nan"), float("nan")

    mean_value = statistics.mean(valid_values)

    if number == 1:
        return mean_value, mean_value

    std_value = statistics.stdev(valid_values)
    standard_error = std_value / math.sqrt(number)

    # 常用自由度对应的双侧95% t临界值
    t_critical_table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        29: 2.045,
    }

    degrees_of_freedom = number - 1
    t_critical = t_critical_table.get(
        degrees_of_freedom,
        1.96,
    )

    margin = t_critical * standard_error

    return mean_value - margin, mean_value + margin


def get_model_information(
    model: YOLO,
    best_weight_path: Path,
) -> dict[str, float]:
    """计算参数量、GFLOPs和权重文件大小。"""

    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    gflops = get_flops(
        model.model,
        imgsz=IMAGE_SIZE,
    )

    weight_size_mb = (
        best_weight_path.stat().st_size / 1024**2
    )

    return {
        "parameters": float(parameter_count),
        "parameters_million": parameter_count / 1_000_000,
        "gflops": safe_float(gflops),
        "weight_size_mb": weight_size_mb,
    }


def train_one_model(
    model_path: Path,
    seed: int,
) -> Path:
    """训练一个模型，并返回best.pt路径。"""

    model_name = model_path.stem
    run_name = f"{model_name}_seed{seed}"
    train_dir = TRAIN_ROOT / run_name
    best_weight = train_dir / "weights" / "best.pt"

    if best_weight.exists() and not RETRAIN_EXISTING:
        print(f"\n已有训练结果，跳过训练：{best_weight}")
        return best_weight

    if train_dir.exists() and not RETRAIN_EXISTING:
        raise RuntimeError(
            f"目录已存在但没有找到best.pt：{train_dir}\n"
            "请删除该不完整目录后重新运行，"
            "或将RETRAIN_EXISTING设置为True。"
        )

    print("\n" + "=" * 70)
    print(f"开始训练：{model_name}")
    print(f"随机种子：{seed}")
    print(f"输出目录：{train_dir}")
    print("=" * 70)

    model = YOLO(str(model_path))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,

        seed=seed,
        deterministic=DETERMINISTIC,
        patience=PATIENCE,
        amp=AMP,

        project=str(TRAIN_ROOT),
        name=run_name,
        exist_ok=RETRAIN_EXISTING,

        pretrained=True,
        resume=False,
        plots=True,
        save=True,
        verbose=True,

        **AUGMENTATION_ARGS,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not best_weight.exists():
        raise FileNotFoundError(
            f"训练结束后没有找到best.pt：{best_weight}"
        )

    return best_weight


def test_one_model(
    original_model_path: Path,
    best_weight_path: Path,
    seed: int,
) -> dict[str, Any]:
    """使用固定test集测试best.pt。"""

    model_name = original_model_path.stem
    run_name = f"{model_name}_seed{seed}"

    print("\n" + "-" * 70)
    print(f"开始测试：{run_name}")
    print(f"测试权重：{best_weight_path}")
    print("-" * 70)

    model = YOLO(str(best_weight_path))

    start_time = time.perf_counter()

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        project=str(TEST_ROOT),
        name=run_name,
        exist_ok=True,

        plots=True,
        save_json=False,
        verbose=True,
    )

    total_test_seconds = time.perf_counter() - start_time

    model_info = get_model_information(
        model=model,
        best_weight_path=best_weight_path,
    )

    speed = getattr(metrics, "speed", {}) or {}

    preprocess_ms = safe_float(
        speed.get("preprocess", float("nan"))
    )
    inference_ms = safe_float(
        speed.get("inference", float("nan"))
    )
    loss_ms = safe_float(
        speed.get("loss", float("nan"))
    )
    postprocess_ms = safe_float(
        speed.get("postprocess", float("nan"))
    )

    if inference_ms > 0 and not math.isnan(inference_ms):
        inference_fps = 1000.0 / inference_ms
    else:
        inference_fps = float("nan")

    result = {
        "model": model_name,
        "seed": seed,
        "best_weight": str(best_weight_path),

        "precision": safe_float(metrics.box.mp),
        "recall": safe_float(metrics.box.mr),
        "map50": safe_float(metrics.box.map50),
        "map75": safe_float(metrics.box.map75),
        "map50_95": safe_float(metrics.box.map),

        "parameters": model_info["parameters"],
        "parameters_million": model_info[
            "parameters_million"
        ],
        "gflops": model_info["gflops"],
        "weight_size_mb": model_info["weight_size_mb"],

        "preprocess_ms_per_image": preprocess_ms,
        "inference_ms_per_image": inference_ms,
        "loss_ms_per_image": loss_ms,
        "postprocess_ms_per_image": postprocess_ms,
        "inference_fps": inference_fps,

        "total_test_seconds": total_test_seconds,
    }

    print("\n本次测试结果：")
    print(f"Precision：{result['precision']:.6f}")
    print(f"Recall：{result['recall']:.6f}")
    print(f"mAP50：{result['map50']:.6f}")
    print(f"mAP75：{result['map75']:.6f}")
    print(f"mAP50-95：{result['map50_95']:.6f}")
    print(
        f"参数量："
        f"{result['parameters_million']:.4f} M"
    )
    print(f"GFLOPs：{result['gflops']:.4f}")
    print(
        f"权重大小："
        f"{result['weight_size_mb']:.4f} MB"
    )
    print(
        f"纯推理时间："
        f"{result['inference_ms_per_image']:.4f} ms/image"
    )
    print(
        f"纯推理FPS："
        f"{result['inference_fps']:.4f}"
    )

    del model
    del metrics
    gc.collect()
    torch.cuda.empty_cache()

    return result


def write_all_results(results: list[dict[str, Any]]) -> Path:
    """保存每个模型、每个随机种子的测试结果。"""

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / "all_runs_results.csv"

    if not results:
        raise RuntimeError("没有可保存的实验结果。")

    fieldnames = list(results[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    return output_path


def write_summary(results: list[dict[str, Any]]) -> Path:
    """按模型计算均值、标准差和95%置信区间。"""

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / "model_summary.csv"

    metric_names = [
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
        "parameters_million",
        "gflops",
        "weight_size_mb",
        "inference_ms_per_image",
        "inference_fps",
    ]

    model_names = sorted(
        {str(result["model"]) for result in results}
    )

    summary_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        model_results = [
            result
            for result in results
            if result["model"] == model_name
        ]

        summary: dict[str, Any] = {
            "model": model_name,
            "runs": len(model_results),
            "seeds": ",".join(
                str(result["seed"])
                for result in model_results
            ),
        }

        for metric_name in metric_names:
            values = [
                safe_float(result[metric_name])
                for result in model_results
            ]

            valid_values = [
                value
                for value in values
                if not math.isnan(value)
            ]

            if not valid_values:
                summary[f"{metric_name}_mean"] = float("nan")
                summary[f"{metric_name}_sd"] = float("nan")
                summary[f"{metric_name}_ci95_low"] = float("nan")
                summary[f"{metric_name}_ci95_high"] = float("nan")
                continue

            mean_value = statistics.mean(valid_values)

            if len(valid_values) >= 2:
                sd_value = statistics.stdev(valid_values)
            else:
                sd_value = 0.0

            ci_low, ci_high = calculate_ci95(valid_values)

            summary[f"{metric_name}_mean"] = mean_value
            summary[f"{metric_name}_sd"] = sd_value
            summary[f"{metric_name}_ci95_low"] = ci_low
            summary[f"{metric_name}_ci95_high"] = ci_high

        summary_rows.append(summary)

    if not summary_rows:
        raise RuntimeError("没有可汇总的实验结果。")

    fieldnames = list(summary_rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return output_path


# ============================================================
# 5. 主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test the YOLO baseline models.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Required random seeds, e.g. --seeds 0 or --seeds 0 1 2 3.")
    parser.add_argument("--tag", help="Optional output folder tag; defaults to the supplied seeds.")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique, nonnegative integers.")
    args.tag = args.tag or "seeds_" + "_".join(map(str, args.seeds))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag may contain only letters, digits, underscores, and hyphens.")
    return args


def main() -> None:
    global TRAIN_ROOT, TEST_ROOT, SUMMARY_ROOT
    args = parse_args()
    seeds = args.seeds
    TRAIN_ROOT = BASE_DIR / "runs_batch_train" / args.tag
    TEST_ROOT = BASE_DIR / "runs_batch_test" / args.tag
    SUMMARY_ROOT = BASE_DIR / "batch_summary" / args.tag
    check_environment()
    save_experiment_config(seeds)

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    total_jobs = len(MODEL_PATHS) * len(seeds)
    current_job = 0

    for model_path in MODEL_PATHS:
        for seed in seeds:
            current_job += 1

            print("\n" + "#" * 70)
            print(
                f"总进度：{current_job}/{total_jobs}"
            )
            print(
                f"模型：{model_path.name}，随机种子：{seed}"
            )
            print("#" * 70)

            try:
                best_weight = train_one_model(
                    model_path=model_path,
                    seed=seed,
                )

                result = test_one_model(
                    original_model_path=model_path,
                    best_weight_path=best_weight,
                    seed=seed,
                )

                all_results.append(result)

                # 每完成一次就保存，避免中途停止时数据丢失
                write_all_results(all_results)
                write_summary(all_results)

            except Exception as error:
                print("\n实验运行失败：")
                print(
                    f"模型：{model_path.name}，"
                    f"随机种子：{seed}"
                )
                print(f"错误类型：{type(error).__name__}")
                print(f"错误内容：{error}")

                # 保存此前已经完成的结果
                if all_results:
                    write_all_results(all_results)
                    write_summary(all_results)

                raise

    all_results_path = write_all_results(all_results)
    summary_path = write_summary(all_results)

    print("\n" + "=" * 70)
    print("全部训练与测试完成。")
    print("逐次实验结果：", all_results_path)
    print("模型统计汇总：", summary_path)
    print("训练结果目录：", TRAIN_ROOT)
    print("测试结果目录：", TEST_ROOT)
    print("=" * 70)


if __name__ == "__main__":
    main()
