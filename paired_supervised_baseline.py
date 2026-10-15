"""Audit paired MyoPS images and train a direct LGE-to-target U-Net baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf

from myocardial_data import (
    TARGET_MODALITIES,
    PairedCase,
    find_paired_cases,
    load_aligned_pair,
    paired_slice_count,
)


def build_supervised_unet(image_size: int = 96) -> tf.keras.Model:
    """Build a compact direct image-to-image U-Net for normalized MyoPS slices."""
    if image_size <= 0 or image_size % 8:
        raise ValueError("image_size must be a positive multiple of 8")

    inputs = tf.keras.layers.Input((image_size, image_size, 1), name="LGE")
    skips: list[tf.Tensor] = []
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    for filters, skip in zip((64, 32, 16), reversed(skips)):
        x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="tanh", name="target")(x)
    return tf.keras.Model(inputs, outputs, name="paired_supervised_unet")


def _slice_generator(
    records: Sequence[PairedCase], image_size: int
):
    for record in records:
        _, lge, target = load_aligned_pair(record)
        for slice_index in range(lge.shape[2]):
            lge_slice = tf.image.resize(
                lge[:, :, slice_index, np.newaxis], (image_size, image_size)
            ).numpy()
            target_slice = tf.image.resize(
                target[:, :, slice_index, np.newaxis], (image_size, image_size)
            ).numpy()
            yield lge_slice.astype(np.float32), target_slice.astype(np.float32)


def make_supervised_dataset(
    records: Sequence[PairedCase],
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    if not records:
        raise ValueError("Cannot create a dataset from an empty case list")
    dataset = tf.data.Dataset.from_generator(
        lambda: _slice_generator(records, image_size),
        output_signature=(
            tf.TensorSpec((image_size, image_size, 1), tf.float32),
            tf.TensorSpec((image_size, image_size, 1), tf.float32),
        ),
    )
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(1024, seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


def _representative_records(
    records: Sequence[PairedCase], limit: int
) -> list[PairedCase]:
    ordered = sorted(records, key=lambda r: (r.center, r.case_id))
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, limit, dtype=int)
    return [ordered[index] for index in indices]


def audit_alignment(
    records: Sequence[PairedCase], output_dir: Path, cases_per_modality: int = 8
) -> list[dict[str, object]]:
    """Save geometry metadata and multi-case LGE/target slice comparison panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for modality in TARGET_MODALITIES:
        subset = [r for r in records if r.target_modality == modality]
        selected = _representative_records(subset, cases_per_modality)
        if not selected:
            print(f"No paired validation cases found for {modality}; skipping audit.")
            continue

        columns = 3
        fig, axes = plt.subplots(
            len(selected), columns, figsize=(columns * 4, len(selected) * 2.7)
        )
        axes = np.asarray(axes).reshape(len(selected), columns)
        for axis in axes.flat:
            axis.axis("off")

        for index, record in enumerate(selected):
            lge_raw = nib.as_closest_canonical(nib.load(str(record.lge_path)))
            target_raw = nib.as_closest_canonical(nib.load(str(record.target_path)))
            _, lge, target = load_aligned_pair(record)
            slice_index = lge.shape[2] // 2
            lge_slice = lge[:, :, slice_index]
            target_slice = target[:, :, slice_index]
            axes[index, 0].imshow(lge_slice, cmap="gray", vmin=-1, vmax=1)
            axes[index, 1].imshow(target_slice, cmap="gray", vmin=-1, vmax=1)
            overlay = np.zeros((*lge_slice.shape, 3), dtype=np.float32)
            overlay[..., 0] = (lge_slice + 1.0) / 2.0
            overlay[..., 1] = (target_slice + 1.0) / 2.0
            axes[index, 2].imshow(np.clip(overlay, 0.0, 1.0))
            axes[index, 0].text(
                0.02,
                0.96,
                f"{record.center} {record.case_id}",
                color="yellow",
                fontsize=8,
                va="top",
                transform=axes[index, 0].transAxes,
            )
            if index == 0:
                for axis, title in zip(
                    axes[index],
                    ("LGE", f"{modality} target", "Overlay: red=LGE, green=target"),
                ):
                    axis.set_title(title)

            affine_delta = float(
                np.max(np.abs(lge_raw.affine - target_raw.affine))
            )
            rows.append(
                {
                    "modality": modality,
                    "center": record.center,
                    "case_id": record.case_id,
                    "lge_shape": "x".join(map(str, lge_raw.shape)),
                    "target_shape": "x".join(map(str, target_raw.shape)),
                    "lge_voxel_sizes_mm": ";".join(
                        f"{size:.4g}" for size in nib.affines.voxel_sizes(lge_raw.affine)
                    ),
                    "target_voxel_sizes_mm": ";".join(
                        f"{size:.4g}"
                        for size in nib.affines.voxel_sizes(target_raw.affine)
                    ),
                    "max_abs_affine_difference": f"{affine_delta:.7g}",
                    "target_resampled_to_lge_grid": not (
                        target_raw.shape == lge_raw.shape
                        and np.allclose(
                            target_raw.affine, lge_raw.affine, rtol=0.0, atol=1e-4
                        )
                    ),
                    "lge_path": str(record.lge_path),
                    "target_path": str(record.target_path),
                }
            )

        fig.suptitle(
            f"Validation registration review: LGE/{modality}; central slices",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"alignment_{modality}.png", dpi=150)
        plt.close(fig)

    if not rows:
        raise ValueError("No paired validation images were available for alignment review")
    with (output_dir / "alignment_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _evaluate_case(
    model: tf.keras.Model,
    record: PairedCase,
    image_size: int,
    training_mean: float,
) -> dict[str, float | str | int]:
    _, lge, target = load_aligned_pair(record)
    lge_batch = tf.image.resize(
        np.moveaxis(lge, 2, 0)[..., np.newaxis], (image_size, image_size)
    )
    target_batch = tf.image.resize(
        np.moveaxis(target, 2, 0)[..., np.newaxis], (image_size, image_size)
    )
    prediction = model.predict(lge_batch, verbose=0)
    prediction = tf.clip_by_value(prediction, -1.0, 1.0)

    identity = tf.clip_by_value(lge_batch, -1.0, 1.0)
    mean_prediction = tf.ones_like(target_batch) * training_mean

    def metrics(candidate: tf.Tensor) -> tuple[float, float, float]:
        mae = tf.reduce_mean(tf.abs(candidate - target_batch))
        ssim = tf.reduce_mean(
            tf.image.ssim(candidate, target_batch, max_val=2.0)
        )
        psnr = tf.reduce_mean(
            tf.image.psnr(candidate, target_batch, max_val=2.0)
        )
        return float(mae), float(ssim), float(psnr)

    result: dict[str, float | str | int] = {
        "modality": record.target_modality,
        "center": record.center,
        "case_id": record.case_id,
        "slice_count": int(target.shape[2]),
    }
    for name, candidate in (
        ("unet", prediction),
        ("lge_copy", identity),
        ("training_mean", mean_prediction),
    ):
        mae, ssim, psnr = metrics(candidate)
        result[f"{name}_mae"] = mae
        result[f"{name}_ssim"] = ssim
        result[f"{name}_psnr"] = psnr
    return result


def _training_target_mean(records: Sequence[PairedCase], image_size: int) -> float:
    total = 0.0
    count = 0
    for _, target in _slice_generator(records, image_size):
        total += float(np.sum(target))
        count += int(target.size)
    if count == 0:
        raise ValueError("Cannot calculate a training mean from an empty dataset")
    return total / count


def _save_prediction_panel(
    model: tf.keras.Model,
    records: Sequence[PairedCase],
    modality: str,
    output_dir: Path,
    image_size: int,
) -> None:
    selected = _representative_records(records, 8)
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    axes = np.asarray(axes).reshape(len(selected), 3)
    for row_index, record in enumerate(selected):
        _, lge, target = load_aligned_pair(record)
        slice_index = lge.shape[2] // 2
        source = tf.image.resize(
            lge[:, :, slice_index, np.newaxis], (image_size, image_size)
        )
        prediction = model.predict(source[np.newaxis, ...], verbose=0)[0, ..., 0]
        prediction = tf.image.resize(
            prediction[..., np.newaxis], lge.shape[:2]
        )[..., 0].numpy()
        for axis, image, title in zip(
            axes[row_index],
            (lge[:, :, slice_index], target[:, :, slice_index], prediction),
            (
                "LGE input",
                f"Reference {modality}",
                f"U-Net {record.center} {record.case_id}",
            ),
        ):
            axis.imshow(image, cmap="gray", vmin=-1, vmax=1)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"predictions_{modality}.png", dpi=150)
    plt.close(fig)


def train_and_evaluate_modality(
    train_records: Sequence[PairedCase],
    valid_records: Sequence[PairedCase],
    modality: str,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    epochs: int,
) -> list[dict[str, float | str | int]]:
    modality_train = [r for r in train_records if r.target_modality == modality]
    modality_valid = [r for r in valid_records if r.target_modality == modality]
    if not modality_train or not modality_valid:
        raise ValueError(
            f"{modality} requires paired train and validation cases; "
            f"found {len(modality_train)} train and {len(modality_valid)} validation"
        )

    train_data = make_supervised_dataset(
        modality_train, image_size, batch_size, shuffle=True
    )
    valid_data = make_supervised_dataset(
        modality_valid, image_size, batch_size
    )
    model = build_supervised_unet(image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model_path = str(output_dir / f"lge_to_{modality}_best.keras")
    history_path = output_dir / f"lge_to_{modality}_history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
    history = model.fit(
        train_data,
        validation_data=valid_data,
        steps_per_epoch=math.ceil(
            paired_slice_count(modality_train) / batch_size
        ),
        validation_steps=math.ceil(
            paired_slice_count(modality_valid) / batch_size
        ),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history.history["loss"], label="Training")
    axis.plot(history.history["val_loss"], label="Validation")
    axis.set(title=f"LGE-to-{modality} supervised U-Net", xlabel="Epoch", ylabel="Huber loss")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"training_{modality}.png", dpi=150)
    plt.close(fig)

    best_model = tf.keras.models.load_model(model_path)
    training_mean = _training_target_mean(modality_train, image_size)
    results = [
        _evaluate_case(best_model, record, image_size, training_mean)
        for record in modality_valid
    ]
    _save_prediction_panel(
        best_model, modality_valid, modality, output_dir, image_size
    )
    return results


def _write_metrics(
    rows: Sequence[dict[str, float | str | int]], output_path: Path
) -> None:
    if not rows:
        raise ValueError("No validation metrics were computed")
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_summary(
    rows: Sequence[dict[str, float | str | int]], output_path: Path
) -> list[dict[str, float | str | int]]:
    summaries: list[dict[str, float | str | int]] = []
    for modality in TARGET_MODALITIES:
        modality_rows = [row for row in rows if row["modality"] == modality]
        if not modality_rows:
            continue
        for method in ("unet", "lge_copy", "training_mean"):
            summaries.append(
                {
                    "modality": modality,
                    "method": method,
                    "mean_case_mae": float(
                        np.mean([row[f"{method}_mae"] for row in modality_rows])
                    ),
                    "mean_case_ssim": float(
                        np.mean([row[f"{method}_ssim"] for row in modality_rows])
                    ),
                    "mean_case_psnr": float(
                        np.mean([row[f"{method}_psnr"] for row in modality_rows])
                    ),
                    "validation_cases": len(modality_rows),
                }
            )
    _write_metrics(summaries, output_path)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing Myo_train and Myo_val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "experiments"
            / "02_supervised"
            / "results"
        ),
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cases-per-modality", type=int, default=8)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Generate the multi-case alignment report without training models",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=sorted(TARGET_MODALITIES),
        default=["T2", "C0"],
        help="Targets to fit independently",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.cases_per_modality <= 0:
        parser.error("epochs, batch-size, and cases-per-modality must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = find_paired_cases(args.data_root / "Myo_train" / "MyoPS_train")
    valid_records = find_paired_cases(args.data_root / "Myo_val" / "MyoPS_val")
    print(
        "Training paired cases:",
        {m: sum(r.target_modality == m for r in train_records) for m in TARGET_MODALITIES},
    )
    print(
        "Validation paired cases:",
        {m: sum(r.target_modality == m for r in valid_records) for m in TARGET_MODALITIES},
    )
    audit_rows = audit_alignment(
        valid_records, args.output_dir, args.cases_per_modality
    )
    print(
        f"Saved alignment review for {len(audit_rows)} validation pairs to "
        f"{args.output_dir}"
    )
    if args.audit_only:
        return

    metric_rows: list[dict[str, float | str | int]] = []
    for modality in args.modalities:
        print(f"\nTraining direct LGE-to-{modality} baseline")
        metric_rows.extend(
            train_and_evaluate_modality(
                train_records,
                valid_records,
                modality,
                args.output_dir,
                args.image_size,
                args.batch_size,
                args.epochs,
            )
        )
    metrics_path = args.output_dir / "validation_metrics_by_case.csv"
    _write_metrics(metric_rows, metrics_path)
    summary_path = args.output_dir / "validation_metrics_summary.csv"
    summaries = _write_metric_summary(metric_rows, summary_path)
    for row in summaries:
        print(
            f"{row['modality']} {row['method']}: "
            f"MAE={row['mean_case_mae']:.4f}, "
            f"SSIM={row['mean_case_ssim']:.4f}, "
            f"PSNR={row['mean_case_psnr']:.2f} dB "
            f"({row['validation_cases']} cases)"
        )
    print(f"Saved per-case validation metrics to {metrics_path}")
    print(f"Saved equal-case metric summary to {summary_path}")


if __name__ == "__main__":
    main()
