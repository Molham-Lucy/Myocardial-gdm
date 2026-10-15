"""Paired MyoPS NIfTI loading for conditional myocardial image synthesis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Sequence

import nibabel as nib
import numpy as np
import tensorflow as tf
from nibabel.processing import resample_from_to


TARGET_MODALITIES = {"C0": 0, "T2": 1}
_CASE_FILE = re.compile(r"(?P<case>Case\d+)_(?P<modality>LGE|C0|T2)\.nii(?:\.gz)?$")


@dataclass(frozen=True)
class PairedCase:
    case_id: str
    center: str
    lge_path: Path
    target_path: Path
    target_modality: str

    @property
    def modality_id(self) -> int:
        return TARGET_MODALITIES[self.target_modality]


def find_paired_cases(root: str | Path) -> list[PairedCase]:
    """Find only within-case LGE/C0 and LGE/T2 pairs under one MyoPS split."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MyoPS split directory does not exist: {root}")

    grouped: dict[tuple[Path, str], dict[str, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = _CASE_FILE.fullmatch(path.name)
        if match is None:
            continue
        case_id = match.group("case")
        grouped.setdefault((path.parent, case_id), {})[match.group("modality")] = path

    paired_cases: list[PairedCase] = []
    for (case_dir, case_id), modalities in sorted(grouped.items()):
        lge_path = modalities.get("LGE")
        if lge_path is None:
            continue
        relative_parts = case_dir.relative_to(root).parts
        center = relative_parts[0] if relative_parts else "UnknownCenter"
        for target_modality in TARGET_MODALITIES:
            target_path = modalities.get(target_modality)
            if target_path is not None:
                paired_cases.append(
                    PairedCase(
                        case_id=case_id,
                        center=center,
                        lge_path=lge_path,
                        target_path=target_path,
                        target_modality=target_modality,
                    )
                )

    if not paired_cases:
        raise ValueError(f"No paired LGE/C0 or LGE/T2 images found under {root}")
    return paired_cases


def normalize_volume(volume: np.ndarray, path: str | Path) -> np.ndarray:
    path = Path(path)
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Image contains non-finite values: {path}")
    foreground = volume[volume != 0]
    if foreground.size == 0:
        raise ValueError(f"Image contains no non-zero voxels: {path}")
    low, high = np.percentile(foreground, (1.0, 99.0))
    if high <= low:
        raise ValueError(f"Image has no usable intensity range: {path}")
    volume = np.clip(volume, low, high)
    return (2.0 * (volume - low) / (high - low) - 1.0).astype(np.float32)


def _slice_axis(image: nib.Nifti1Image, volume_shape: Sequence[int]) -> int:
    singleton_axes = [
        axis for axis, size in enumerate(volume_shape) if size == 1
    ]
    if len(singleton_axes) == 1:
        return singleton_axes[0]
    return int(np.argmax(nib.affines.voxel_sizes(image.affine)))


def _move_slice_axis_last(image: nib.Nifti1Image, volume: np.ndarray) -> np.ndarray:
    return np.moveaxis(volume, _slice_axis(image, volume.shape), 2)


def load_aligned_pair(record: PairedCase) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Load a paired case, resampling its target onto the LGE voxel grid if needed."""
    lge_img = nib.as_closest_canonical(nib.load(str(record.lge_path)))
    target_img = nib.as_closest_canonical(nib.load(str(record.target_path)))
    if len(lge_img.shape) != 3 or len(target_img.shape) != 3:
        raise ValueError(
            f"Expected 3D MyoPS volumes for {record.case_id}; "
            f"got LGE {lge_img.shape}, {record.target_modality} {target_img.shape}"
        )

    if target_img.shape != lge_img.shape or not np.allclose(
        target_img.affine, lge_img.affine, rtol=0.0, atol=1e-4
    ):
        target_img = resample_from_to(
            target_img, (lge_img.shape, lge_img.affine), order=1
        )

    lge = normalize_volume(lge_img.get_fdata(dtype=np.float32), record.lge_path)
    target = normalize_volume(
        target_img.get_fdata(dtype=np.float32), record.target_path
    )
    lge = _move_slice_axis_last(lge_img, lge)
    target = _move_slice_axis_last(lge_img, target)
    return lge_img, lge, target


def load_lge_volume(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load and robustly normalize one 3D LGE volume for conditional synthesis."""
    path = Path(path)
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D LGE volume, got {image.shape}: {path}")
    data = normalize_volume(image.get_fdata(dtype=np.float32), path)
    return image, _move_slice_axis_last(image, data)


def _iter_paired_slices(
    records: Sequence[PairedCase],
) -> Iterator[tuple[np.ndarray, np.ndarray, np.int32]]:
    for record in records:
        _, lge, target = load_aligned_pair(record)
        for slice_index in range(lge.shape[2]):
            yield (
                lge[:, :, slice_index],
                target[:, :, slice_index],
                np.int32(record.modality_id),
            )


def make_paired_dataset(
    records: Sequence[PairedCase],
    alpha_cumprod: np.ndarray,
    timesteps: int,
    image_size: int = 128,
    batch_size: int = 16,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a streamed 2D-slice dataset with LGE and target-modality conditioning."""
    if not records:
        raise ValueError("Cannot build a dataset from an empty paired-case list")
    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image_size and batch_size must be positive")
    if len(alpha_cumprod) != timesteps + 1:
        raise ValueError("alpha_cumprod must contain entries for steps 0..timesteps")

    dataset = tf.data.Dataset.from_generator(
        lambda: _iter_paired_slices(records),
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    def resize_pair(
        lge: tf.Tensor, target: tf.Tensor, modality: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        lge = tf.image.resize(lge[..., tf.newaxis], (image_size, image_size))
        target = tf.image.resize(
            target[..., tf.newaxis], (image_size, image_size)
        )
        return lge, target, modality

    dataset = dataset.map(
        resize_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1024, seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)

    alpha_cumprod_tensor = tf.constant(alpha_cumprod, dtype=tf.float32)

    def add_noise(
        lge: tf.Tensor, target: tf.Tensor, modality: tf.Tensor
    ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        batch_shape = tf.shape(target)
        time = tf.random.uniform(
            (batch_shape[0],), minval=1, maxval=timesteps + 1, dtype=tf.int32
        )
        alpha_bar = tf.gather(alpha_cumprod_tensor, time)
        alpha_bar = tf.reshape(alpha_bar, (-1, 1, 1, 1))
        noise = tf.random.normal(batch_shape)
        noisy_target = (
            tf.sqrt(alpha_bar) * target + tf.sqrt(1.0 - alpha_bar) * noise
        )
        return {
            "X_noisy": noisy_target,
            "X_condition": lge,
            "time": time,
            "modality": modality,
        }, noise

    return dataset.map(
        add_noise, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def paired_case_counts(records: Sequence[PairedCase]) -> Counter[str]:
    """Count paired patient cases, rather than their 2D slices, by target modality."""
    return Counter(
        {
            modality: len(
                {
                    (record.center, record.case_id)
                    for record in records
                    if record.target_modality == modality
                }
            )
            for modality in TARGET_MODALITIES
        }
    )


class _TimeEncoding(tf.keras.layers.Layer):
    def __init__(self, timesteps: int, embedding_size: int) -> None:
        super().__init__(dtype=tf.float32)
        if embedding_size % 2:
            raise ValueError("embedding_size must be even")
        steps, dimensions = np.meshgrid(
            np.arange(timesteps + 1), 2 * np.arange(embedding_size // 2)
        )
        encodings = np.empty((timesteps + 1, embedding_size), dtype=np.float32)
        encodings[:, ::2] = np.sin(
            steps / 10_000 ** (dimensions / embedding_size)
        ).T
        encodings[:, 1::2] = np.cos(
            steps / 10_000 ** (dimensions / embedding_size)
        ).T
        self.encodings = tf.constant(encodings)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.encodings, inputs)


def build_conditional_diffusion_model(
    timesteps: int, image_size: int = 128, embedding_size: int = 64
) -> tf.keras.Model:
    """Build a 2D U-Net that predicts target noise from LGE and target modality."""
    if image_size <= 0 or image_size % 8:
        raise ValueError("image_size must be a positive multiple of 8")

    noisy_input = tf.keras.layers.Input(
        shape=(image_size, image_size, 1), name="X_noisy"
    )
    lge_input = tf.keras.layers.Input(
        shape=(image_size, image_size, 1), name="X_condition"
    )
    time_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="time")
    modality_input = tf.keras.layers.Input(
        shape=(), dtype=tf.int32, name="modality"
    )

    time_encoding = _TimeEncoding(timesteps, embedding_size)(time_input)
    modality_encoding = tf.keras.layers.Embedding(
        len(TARGET_MODALITIES), embedding_size
    )(modality_input)
    condition_encoding = tf.keras.layers.Add()(
        [time_encoding, modality_encoding]
    )

    z = tf.keras.layers.Concatenate()([noisy_input, lge_input])
    z = tf.keras.layers.Conv2D(32, 3, padding="same")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    time = tf.keras.layers.Dense(32)(condition_encoding)
    z = z + time[:, tf.newaxis, tf.newaxis, :]

    skip = z
    cross_skips = []
    for filters in (64, 128, 256):
        z = tf.keras.layers.Activation("relu")(z)
        z = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Activation("relu")(z)
        z = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(z)
        z = tf.keras.layers.BatchNormalization()(z)

        cross_skips.append(z)
        z = tf.keras.layers.MaxPooling2D(2)(z)
        skip_link = tf.keras.layers.Conv2D(
            filters, 1, strides=2, padding="same"
        )(skip)
        z = tf.keras.layers.Add()([z, skip_link])
        time = tf.keras.layers.Dense(filters)(condition_encoding)
        z = z + time[:, tf.newaxis, tf.newaxis, :]
        skip = z

    for filters in (128, 64, 32):
        z = tf.keras.layers.Activation("relu")(z)
        z = tf.keras.layers.Conv2DTranspose(
            filters, 3, padding="same"
        )(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Activation("relu")(z)
        z = tf.keras.layers.Conv2DTranspose(
            filters, 3, padding="same"
        )(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.UpSampling2D(2)(z)

        skip_link = tf.keras.layers.UpSampling2D(2)(skip)
        skip_link = tf.keras.layers.Conv2D(filters, 1, padding="same")(
            skip_link
        )
        z = tf.keras.layers.Add()([z, skip_link])
        time = tf.keras.layers.Dense(filters)(condition_encoding)
        z = z + time[:, tf.newaxis, tf.newaxis, :]
        z = tf.keras.layers.Concatenate()([z, cross_skips.pop()])
        skip = z

    noise_prediction = tf.keras.layers.Conv2D(1, 3, padding="same")(z)
    return tf.keras.Model(
        inputs={
            "X_noisy": noisy_input,
            "X_condition": lge_input,
            "time": time_input,
            "modality": modality_input,
        },
        outputs=noise_prediction,
    )


def generate_conditioned(
    model: tf.keras.Model,
    lge_batch: tf.Tensor,
    target_modality: str,
    alpha_cumprod: np.ndarray,
    timesteps: int,
    sampling_steps: int = 100,
) -> tf.Tensor:
    """Generate target slices with deterministic DDIM sampling."""
    if target_modality not in TARGET_MODALITIES:
        raise ValueError(
            f"target_modality must be one of {tuple(TARGET_MODALITIES)}"
        )
    if sampling_steps <= 0 or sampling_steps > timesteps:
        raise ValueError("sampling_steps must be in the range 1..timesteps")
    if len(alpha_cumprod) != timesteps + 1:
        raise ValueError("alpha_cumprod must contain entries for steps 0..timesteps")

    lge_batch = tf.convert_to_tensor(lge_batch, dtype=tf.float32)
    if lge_batch.shape.rank != 4:
        raise ValueError("lge_batch must have shape [batch, height, width, channels]")
    batch_size = tf.shape(lge_batch)[0]
    modality = tf.fill(
        (batch_size,), tf.cast(TARGET_MODALITIES[target_modality], tf.int32)
    )
    image = tf.random.normal(tf.shape(lge_batch), dtype=tf.float32)
    times = np.linspace(timesteps, 1, sampling_steps, dtype=np.int32)
    alpha_cumprod = np.asarray(alpha_cumprod, dtype=np.float32)

    for index, time_step in enumerate(times):
        previous_step = int(times[index + 1]) if index + 1 < len(times) else 0
        time = tf.fill((batch_size,), int(time_step))
        predicted_noise = model(
            {
                "X_noisy": image,
                "X_condition": lge_batch,
                "time": time,
                "modality": modality,
            },
            training=False,
        )
        alpha_bar = alpha_cumprod[time_step]
        previous_alpha_bar = (
            alpha_cumprod[previous_step] if previous_step else 1.0
        )
        predicted_clean = (
            image - np.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / np.sqrt(alpha_bar)
        image = (
            np.sqrt(previous_alpha_bar) * predicted_clean
            + np.sqrt(1.0 - previous_alpha_bar) * predicted_noise
        )
    return tf.clip_by_value(image, -1.0, 1.0)


def synthesize_volume(
    model: tf.keras.Model,
    lge_path: str | Path,
    target_modality: str,
    alpha_cumprod: np.ndarray,
    timesteps: int,
    image_size: int = 128,
    sampling_steps: int = 100,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize one 3D target volume on the canonical LGE voxel grid."""
    image, lge = load_lge_volume(lge_path)
    height, width, depth = lge.shape
    generated_slices = []
    for slice_index in range(depth):
        lge_slice = tf.image.resize(
            lge[:, :, slice_index, np.newaxis], (image_size, image_size)
        )
        generated = generate_conditioned(
            model,
            lge_slice[tf.newaxis, ...],
            target_modality,
            alpha_cumprod,
            timesteps,
            sampling_steps,
        )
        generated = tf.image.resize(generated, (height, width))[0, ..., 0]
        generated_slices.append(generated.numpy())
    generated_volume = np.stack(generated_slices, axis=2).astype(np.float32)
    slice_axis = _slice_axis(image, image.shape)
    generated_volume = np.moveaxis(generated_volume, 2, slice_axis)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)
        nib.save(
            nib.Nifti1Image(generated_volume, image.affine, header=header),
            str(output_path),
        )
    return generated_volume
