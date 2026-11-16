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

# DDS Head迁移模型：
# 先由自定义YAML构建网络，再加载对应官方模型的预训练权重。
MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "dds_yolo11n",
        "yaml": BASE_DIR / "dds_yolo11n.yaml",
        "pretrained": BASE_DIR / "yolo11n.pt",
    },
    {
        "name": "dds_yolo12n",
        "yaml": BASE_DIR / "dds_yolo12n.yaml",
        "pretrained": BASE_DIR / "yolo12n.pt",
    },
]

# 使用独立输出目录，避免覆盖原始YOLO11n、YOLO12n的训练结果。
TRAIN_ROOT = BASE_DIR / "runs_dds_train"
TEST_ROOT = BASE_DIR / "runs_dds_test"
SUMMARY_ROOT = BASE_DIR / "summary_dds"


# ============================================================
# 2. 训练参数
# ============================================================

EPOCHS = 100
# Seeds are selected from terminal arguments in main().

IMAGE_SIZE = 640
BATCH_SIZE = 4
WORKERS = 4
DEVICE = 0

OPTIMIZER = "SGD"
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

PATIENCE = 100
AMP = True
DETERMINISTIC = True

# False：若同名目录中已经存在best.pt，则跳过该次训练并直接测试。
# True：允许使用同名目录重新训练并覆盖结果。
RETRAIN_EXISTING = False


# ============================================================
# 3. 数据增强参数
# 必须与原始YOLO11n和YOLO12n对比训练保持一致
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

def safe_float(value: Any) -> float:
    """将指标安全转换为浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def check_environment() -> None:
    """检查数据配置、模型文件和CUDA环境。"""

    print("=" * 78)
    print("项目目录：", BASE_DIR)
    print("数据配置：", DATA_YAML)
    print("Python：", sys.version.split()[0])
    print("PyTorch：", torch.__version__)
    print("Ultralytics：", ultralytics.__version__)
    print("Ultralytics路径：", ultralytics.__file__)
    print("PyTorch CUDA：", torch.version.cuda)
    print("CUDA可用：", torch.cuda.is_available())

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"未找到数据集配置文件：{DATA_YAML}")

    missing_files: list[str] = []

    for spec in MODEL_SPECS:
        for key in ("yaml", "pretrained"):
            file_path = Path(spec[key])
            if not file_path.exists():
                missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(
            "以下模型文件不存在：\n" + "\n".join(missing_files)
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不能进行GPU训练。")

    if DEVICE >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU编号{DEVICE}不存在，当前GPU数量为"
            f"{torch.cuda.device_count()}。"
        )

    properties = torch.cuda.get_device_properties(DEVICE)

    print("训练GPU：", torch.cuda.get_device_name(DEVICE))
    print("GPU显存：", f"{properties.total_memory / 1024**3:.2f} GB")
    print("=" * 78)


def save_experiment_config(seeds: list[int]) -> None:
    """保存运行环境和统一实验参数。"""

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_path": ultralytics.__file__,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "gpu_count": torch.cuda.device_count(),
    }

    config = {
        "dataset_yaml": str(DATA_YAML),
        "models": [
            {
                "name": spec["name"],
                "yaml": str(spec["yaml"]),
                "pretrained": str(spec["pretrained"]),
            }
            for spec in MODEL_SPECS
        ],
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
        "resume": False,
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


def calculate_ci95(values: list[float]) -> tuple[float, float]:
    """
    计算均值的双侧95%置信区间。

    使用Student t分布。正式实验为4次运行时，自由度为3，
    临界值为3.182。
    """

    valid_values = [
        value for value in values if not math.isnan(value)
    ]

    n = len(valid_values)

    if n == 0:
        return float("nan"), float("nan")

    mean_value = statistics.mean(valid_values)

    if n == 1:
        return mean_value, mean_value

    sd_value = statistics.stdev(valid_values)
    standard_error = sd_value / math.sqrt(n)

    # 双侧95%置信区间的常用t临界值。
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

    degrees_of_freedom = n - 1
    t_critical = t_critical_table.get(degrees_of_freedom, 1.96)
    margin = t_critical * standard_error

    return mean_value - margin, mean_value + margin


def get_model_information(
    model: YOLO,
    best_weight_path: Path,
) -> dict[str, float]:
    """
    统计未融合模型的参数量、GFLOPs和权重大小。

    该函数应在model.val()之前调用，避免验证阶段可能发生的
    Conv-BN融合导致GFLOPs统计口径变化。
    """

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
    model_spec: dict[str, Any],
    seed: int,
) -> Path:
    """训练一个DDS迁移模型，并返回best.pt路径。"""

    model_name = str(model_spec["name"])
    model_yaml = Path(model_spec["yaml"])
    pretrained_weight = Path(model_spec["pretrained"])

    # 将epoch写入目录名，避免1轮测试结果与100轮正式结果混淆。
    run_name = f"{model_name}_epoch{EPOCHS}_seed{seed}"
    train_dir = TRAIN_ROOT / run_name
    best_weight = train_dir / "weights" / "best.pt"

    if best_weight.exists() and not RETRAIN_EXISTING:
        print(f"\n已有训练结果，跳过训练：{best_weight}")
        return best_weight

    if train_dir.exists() and not RETRAIN_EXISTING:
        raise RuntimeError(
            f"训练目录已存在，但未找到best.pt：{train_dir}\n"
            "请删除该不完整目录，或将RETRAIN_EXISTING改为True。"
        )

    print("\n" + "=" * 78)
    print("开始训练DDS迁移模型")
    print("模型名称：", model_name)
    print("模型YAML：", model_yaml)
    print("预训练权重：", pretrained_weight)
    print("随机种子：", seed)
    print("训练轮数：", EPOCHS)
    print("输出目录：", train_dir)
    print("=" * 78)

    # 第一步：根据DDS模型YAML构建新网络。
    model = YOLO(str(model_yaml))

    # 第二步：加载对应基准模型的预训练权重。
    # 新增DDS层因结构不匹配而随机初始化属于正常现象。
    model.load(str(pretrained_weight))

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

        # 明确禁止从旧last.pt断点续训。
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
            f"训练结束后未找到best.pt：{best_weight}"
        )

    return best_weight


def test_one_model(
    model_name: str,
    best_weight_path: Path,
    seed: int,
) -> dict[str, Any]:
    """使用固定test集评价对应best.pt。"""

    run_name = f"{model_name}_epoch{EPOCHS}_seed{seed}"

    print("\n" + "-" * 78)
    print("开始固定测试集评价")
    print("模型名称：", model_name)
    print("测试权重：", best_weight_path)
    print("-" * 78)

    model = YOLO(str(best_weight_path))

    # 在验证前统计模型信息，统一采用未融合模型口径。
    model_info = get_model_information(
        model=model,
        best_weight_path=best_weight_path,
    )

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
        "epochs": EPOCHS,
        "best_weight": str(best_weight_path),

        "precision": safe_float(metrics.box.mp),
        "recall": safe_float(metrics.box.mr),
        "map50": safe_float(metrics.box.map50),
        "map75": safe_float(metrics.box.map75),
        "map50_95": safe_float(metrics.box.map),

        "parameters": model_info["parameters"],
        "parameters_million": model_info["parameters_million"],
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
    print(f"mAP@0.5：{result['map50']:.6f}")
    print(f"mAP@0.75：{result['map75']:.6f}")
    print(f"mAP@0.5:0.95：{result['map50_95']:.6f}")
    print(
        "参数量："
        f"{result['parameters_million']:.4f} M"
    )
    print(f"GFLOPs：{result['gflops']:.4f}")
    print(
        "权重大小："
        f"{result['weight_size_mb']:.4f} MB"
    )
    print(
        "纯推理时间："
        f"{result['inference_ms_per_image']:.4f} ms/image"
    )
    print(f"相对FPS：{result['inference_fps']:.4f}")

    del model
    del metrics
    gc.collect()
    torch.cuda.empty_cache()

    return result


def write_all_results(
    results: list[dict[str, Any]],
) -> Path:
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


def write_summary(
    results: list[dict[str, Any]],
) -> Path:
    """按模型计算均值、样本标准差和95%置信区间。"""

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
            "epochs": EPOCHS,
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
                summary[f"{metric_name}_ci95_half_width"] = float("nan")
                summary[f"{metric_name}_ci95_low"] = float("nan")
                summary[f"{metric_name}_ci95_high"] = float("nan")
                continue

            mean_value = statistics.mean(valid_values)

            if len(valid_values) >= 2:
                sd_value = statistics.stdev(valid_values)
            else:
                sd_value = 0.0

            ci_low, ci_high = calculate_ci95(valid_values)
            ci_half_width = (ci_high - ci_low) / 2.0

            summary[f"{metric_name}_mean"] = mean_value
            summary[f"{metric_name}_sd"] = sd_value
            summary[f"{metric_name}_ci95_half_width"] = ci_half_width
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
    parser = argparse.ArgumentParser(description="Train and test DDS-head transfer models.")
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
    TRAIN_ROOT = BASE_DIR / "runs_dds_train" / args.tag
    TEST_ROOT = BASE_DIR / "runs_dds_test" / args.tag
    SUMMARY_ROOT = BASE_DIR / "summary_dds" / args.tag
    check_environment()
    save_experiment_config(seeds)

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    total_jobs = len(MODEL_SPECS) * len(seeds)
    current_job = 0

    for model_spec in MODEL_SPECS:
        for seed in seeds:
            current_job += 1

            print("\n" + "#" * 78)
            print(f"总进度：{current_job}/{total_jobs}")
            print(
                f"模型：{model_spec['name']}，随机种子：{seed}"
            )
            print("#" * 78)

            try:
                best_weight = train_one_model(
                    model_spec=model_spec,
                    seed=seed,
                )

                result = test_one_model(
                    model_name=str(model_spec["name"]),
                    best_weight_path=best_weight,
                    seed=seed,
                )

                all_results.append(result)

                # 每完成一次即保存，避免程序中断后已完成结果丢失。
                write_all_results(all_results)
                write_summary(all_results)

            except Exception as error:
                print("\n实验运行失败：")
                print(
                    f"模型：{model_spec['name']}，"
                    f"随机种子：{seed}"
                )
                print(f"错误类型：{type(error).__name__}")
                print(f"错误内容：{error}")

                if all_results:
                    write_all_results(all_results)
                    write_summary(all_results)

                raise

    all_results_path = write_all_results(all_results)
    summary_path = write_summary(all_results)

    print("\n" + "=" * 78)
    print("全部DDS迁移模型训练与测试完成。")
    print("逐次实验结果：", all_results_path)
    print("模型统计汇总：", summary_path)
    print("训练结果目录：", TRAIN_ROOT)
    print("测试结果目录：", TEST_ROOT)
    print("=" * 78)


if __name__ == "__main__":
    main()
