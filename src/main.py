import os
import re
import queue
import time
import torch
import logging
import threading
import numpy as np
import tkinter as tk
import torch.nn as nn
import cv2
from pathlib import Path
from scipy import ndimage
from PIL import Image, ImageTk
from enum import Enum, StrEnum
from typing import Any, Callable
from rich.logging import RichHandler
from matplotlib.figure import Figure
from pytorch_grad_cam import GradCAM
from torchvision import models, transforms
from torchvision.transforms import functional as F
from tkinter import ttk, filedialog, messagebox
from torch.utils.data import DataLoader, Dataset
from dataclasses import asdict, dataclass, field
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# =============================================================
# ENUMS, CLASSES, CONFIGS e LOG
# =============================================================

_rich_handler = RichHandler(
    show_time=True,
    show_path=False,
    markup=True
)

_rich_handler.setFormatter(
    logging.Formatter("%(message)s")
)

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

LOG_LEVEL = LogLevel.INFO

LOGGER = logging.getLogger("PAI")
LOGGER.addHandler(_rich_handler)
LOGGER.setLevel(LOG_LEVEL.value)
LOGGER.propagate = False

class SetType(Enum):
    TRAIN = "train"
    TEST = "test"

class SampleSide(Enum):
    LEFT = "left"
    RIGHT = "right"

class SampleView(Enum):
    CC = "cc"
    MLO = "mlo"

class SampleBiradsClass(StrEnum):
    D = "BIRADS I"
    E = "BIRADS II"
    F = "BIRADS III"
    G = "BIRADS IV"

DEFAULT_BIRADS_LABELS = tuple(birads.value for birads in SampleBiradsClass)

@dataclass(frozen=True, slots=True)
class Sample:
    path: Path
    filename: str
    birads_class: SampleBiradsClass
    number: int
    set_type: SetType
    side: SampleSide
    view: SampleView

class ModelType(StrEnum):
    RESNET18 = "resnet18"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"

@dataclass(slots=True)
class SegmentationConfig:
    threshold_offset: int = 10
    closing_iterations: int = 1
    kernel_size: int = 25
    crop: bool = False

@dataclass(slots=True)
class DatasetConfig:
    segmented: bool = True
    augmentation: bool = False
    binary_classification: bool = True
    image_size: int = 224
    segmentation_config: SegmentationConfig = field(default_factory=SegmentationConfig)

@dataclass(slots=True)
class TrainingConfig:
    model_type: ModelType = ModelType.RESNET18
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    binary_classification: bool = False
    model_path: Path = Path("model.pth")

_DEFAULT_TRAINING_CONFIG = TrainingConfig()
_DEFAULT_SEGMENTATION_CONFIG = SegmentationConfig()

# =============================================================
# DATASET MANAGER
# =============================================================

class DatasetManager:
    IMAGE_EXTENSIONS: set[str] = {".png", ".tif", ".tiff"}
    TEST_SPLIT_MOD: int = 4

    def __init__(self):
        self._dataset_dir: Path | None = None
        self.samples: list[Sample] = []
        self.train_samples: list[Sample] = []
        self.test_samples: list[Sample] = []

    def load_dataset(self, dataset_dir_path: str | Path) -> None:
        self._dataset_dir = Path(dataset_dir_path) if isinstance(dataset_dir_path, str) else dataset_dir_path

        self._validate_directory(self._dataset_dir)
        self._clear_old_data()

        LOGGER.info("Carregando dataset do diretório: %s", self._dataset_dir)
        for file in self._dataset_dir.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            sample = self._create_sample(file)
            LOGGER.debug("Sample criado: %s", sample)
            
            self.samples.append(sample)

            if sample.set_type == SetType.TRAIN:
                self.train_samples.append(sample)
            else:
                self.test_samples.append(sample)

        LOGGER.info("Dataset carregado com sucesso. Total de amostras: %d", len(self.samples))
        LOGGER.info("Amostras de treino: %d", len(self.train_samples))
        LOGGER.info("Amostras de teste: %d", len(self.test_samples))

    def _validate_directory(self, path: Path) -> None:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Diretório inválido: {path}")
    
    def _clear_old_data(self) -> None:
        self.samples.clear()
        self.train_samples.clear()
        self.test_samples.clear()

    def _create_sample(self, file: Path) -> Sample:
        birads_class = self._get_birads_class(file.name)

        number = self._get_image_number(file.name)

        side = self._get_side(file.name)

        view = self._get_view(file.name)

        set_type = SetType.TEST if number % self.TEST_SPLIT_MOD == 0 else SetType.TRAIN

        return Sample(
            path=file,
            filename=file.name,
            birads_class=birads_class,
            number=number,
            set_type=set_type,
            side=side,
            view=view
        )

    def _get_birads_class(self, filename: str) -> SampleBiradsClass:
        return SampleBiradsClass[filename[0].upper()]

    def _get_image_number(self, filename: str) -> int:
        match = re.search(r"\((\d+)\)", filename)

        if not match:
            return 1

        return int(match.group(1))

    def _get_side(self, filename: str) -> SampleSide:
        low = filename.lower()

        if "left" in low:
            return SampleSide.LEFT
        
        if "right" in low:
            return SampleSide.RIGHT
        
        raise ValueError(f"Lado não encontrado no filename: {filename}")

    def _get_view(self, filename: str) -> SampleView:
        up = filename.lower()

        if "cc" in up:
            return SampleView.CC
        
        if "mlo" in up:
            return SampleView.MLO
        
        raise ValueError(f"View não encontrada no filename: {filename}")

# =============================================================
# IMAGE MANAGER
# =============================================================

class ImageManager:
    @staticmethod
    def load(sample: Sample) -> np.ndarray:
        LOGGER.debug("Carregando imagem do sample: %s", sample)
        image = Image.open(sample.path)

        if image.mode != "L":
            image = image.convert("L")

        image_array = np.array(image)

        return image_array

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        LOGGER.debug("Normalizando imagem com shape: %s e dtype: %s", image.shape, image.dtype)

        image = image.astype(np.float32)

        image -= image.min()

        max_value = image.max()

        if max_value > 0:
            image /= max_value

        return image

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        LOGGER.debug("Redimensionando imagem para %dx%d", width, height)

        pil_image = Image.fromarray(image)

        pil_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)

        return np.array(pil_image)
    
    @staticmethod
    def to_pil(image: np.ndarray) -> Image.Image:
        image = image.astype(np.float32)

        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        image = (image * 255).astype(np.uint8)

        return Image.fromarray(image)

# =============================================================
# SEGMENTATION PROCESSOR
# =============================================================

class SegmentationProcessor:
    DOWNSCALE_FACTOR = 0.5

    @staticmethod
    def create_mask(image: np.ndarray, threshold_offset: int = 0) -> np.ndarray:
        LOGGER.debug("Criando máscara de segmentação...")

        threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        threshold = int(threshold) + threshold_offset
        threshold = max(0, min(255, threshold))

        return (image > threshold).astype(np.uint8)

    @staticmethod
    def largest_component(mask: np.ndarray) -> np.ndarray:
        LOGGER.debug("Extraindo maior componente conectada...")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        largest_label = (np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)

        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def refine_mask(mask: np.ndarray, closing_iterations: int = 5, kernel_size: int = 15) -> np.ndarray:
        LOGGER.debug("Refinando máscara...")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask.astype(np.uint8)

    @staticmethod
    def crop_to_bounding_box(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        LOGGER.debug("Aplicando crop...")

        points = cv2.findNonZero(mask)

        if points is None:
            return image

        x, y, w, h = cv2.boundingRect(points)

        return image[y:y + h, x:x + w]

    @staticmethod
    def segment(image: np.ndarray, config: SegmentationConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
        if config is None:
            config = SegmentationConfig()

        start_total = time.perf_counter()

        original_height, original_width = image.shape

        factor = SegmentationProcessor.DOWNSCALE_FACTOR

        if factor < 1.0:
            reduced = cv2.resize(image,
                (int(original_width * factor), int(original_height * factor)), interpolation=cv2.INTER_AREA)
        else:
            reduced = image

        mask = SegmentationProcessor.create_mask(reduced, threshold_offset=config.threshold_offset)
        mask = SegmentationProcessor.largest_component(mask)

        kernel_size = max(3, int(config.kernel_size * factor))

        if kernel_size % 2 == 0:
            kernel_size += 1

        mask = SegmentationProcessor.refine_mask(mask, closing_iterations=config.closing_iterations, kernel_size=kernel_size)

        if factor < 1.0:
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        segmented = image.copy()
        segmented[mask == 0] = 0

        if config.crop:
            segmented = SegmentationProcessor.crop_to_bounding_box(segmented, mask)
            mask = SegmentationProcessor.crop_to_bounding_box(mask, mask)

        elapsed = time.perf_counter() - start_total

        LOGGER.debug("Segmentação concluída em %.3fs", elapsed)

        return mask, segmented
    
# =============================================================
# AUMENTO DE DADOS
# =============================================================

class DataAugmentationProcessor:
    ROTATIONS = [-20, -10, 0, 10, 20]
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        LOGGER.debug("Rotacionando imagem em %.1f graus", angle)

        rotated = ndimage.rotate(image, angle=angle, reshape=False, mode="constant", cval=0)

        return rotated.astype(image.dtype)

# =============================================================
# TRANSFORMAÇÃO
# =============================================================

class MammographyDataset(Dataset):
    BINARY_CLASS_MAPPING = {
        SampleBiradsClass.D: 0,
        SampleBiradsClass.E: 0,
        SampleBiradsClass.F: 1,
        SampleBiradsClass.G: 1
    }

    FOUR_CLASS_MAPPING = {
        SampleBiradsClass.D: 0,
        SampleBiradsClass.E: 1,
        SampleBiradsClass.F: 2,
        SampleBiradsClass.G: 3
    }

    def __init__(self, samples: list[Sample], config: DatasetConfig):
        self.samples = samples
        self.config = config
        self.cache: dict[int, tuple[torch.Tensor, int]] = {}
        self._segmentation_cache: dict[Path, np.ndarray] = {}
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        if self.config.augmentation:
            return len(self.samples) * len(DataAugmentationProcessor.ROTATIONS)
        
        return len(self.samples)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        
        if self.config.augmentation:
            sample_index = index // len(DataAugmentationProcessor.ROTATIONS)
            rotation_index = index % len(DataAugmentationProcessor.ROTATIONS)
            sample = self.samples[sample_index]
            angle = DataAugmentationProcessor.ROTATIONS[rotation_index]
        else:
            sample = self.samples[index]
            angle = 0

        image = ImageManager.load(sample)

        if self.config.segmented:
            if sample.path not in self._segmentation_cache:
                _, segmented = SegmentationProcessor.segment(image, self.config.segmentation_config)
                self._segmentation_cache[sample.path] = segmented
            image = self._segmentation_cache[sample.path]

        image = ImageManager.resize(image, self.config.image_size, self.config.image_size)
        image = ImageManager.normalize(image)
        image = self._to_rgb(image)
        
        tensor_image = self.transform(image)
        
        if angle != 0:
            tensor_image = self._rotate_tensor(tensor_image, angle)

        label = self._get_label(sample)
        result = (tensor_image, torch.tensor(label, dtype=torch.long))

        self.cache[index] = result
        return result

    def _get_label(self, sample: Sample) -> int:
        if self.config.binary_classification:
            return self.BINARY_CLASS_MAPPING[sample.birads_class]
        return self.FOUR_CLASS_MAPPING[sample.birads_class]
    
    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        return image
    
    def _rotate_tensor(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        rotated = F.affine(
            tensor,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[0, 0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
        
        if squeeze:
            rotated = rotated.squeeze(0)
        
        return rotated
        
# =============================================================
# TREINAMENTO, CLASSIFICAÇÃO E AVALIAÇÃO
# =============================================================

class MetricsCalculator:
    @staticmethod
    def binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": cm,
            "report": classification_report(y_true, y_pred, zero_division=0)
        }

    @staticmethod
    def multiclass_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        specificities = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        mean_specificity = np.mean(specificities) if specificities else 0.0

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_macro,
            "sensitivity": recall_macro,
            "mean_sensitivity": recalls.mean() if recalls.size else 0.0,
            "specificity": mean_specificity,
            "mean_specificity": mean_specificity,
            "f1": f1_macro,
            "confusion_matrix": cm,
            "report": classification_report(y_true, y_pred, zero_division=0)
        }

class TrainingManager:
    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: MammographyDataset | None = None, 
        test_dataset: MammographyDataset | None = None, 
        progress_callback: Callable[[int, int, float, float, float, float], None] | None = None,
        batch_progress_callback: Callable[[int, int, int, int], None] | None = None,
    ):
        self.config = config
        self.device = self._get_device()
        self.progress_callback = progress_callback
        self.batch_progress_callback = batch_progress_callback
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        if train_dataset is not None:
            num_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                prefetch_factor=3,
                drop_last=True
            )

        if test_dataset is not None:
            num_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                prefetch_factor=3,
                drop_last=False
            )

        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=config.learning_rate)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        self.cancel_requested = False
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            LOGGER.info("GPU detectada: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda")

        LOGGER.warning("GPU não encontrada. Utilizando CPU.")
        return torch.device("cpu")

    def _create_model(self) -> nn.Module:
        num_classes = 2 if self.config.binary_classification else 4
        dropout_rate = float(self.config.dropout_rate)

        if self.config.model_type == ModelType.RESNET18:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(model.fc.in_features, num_classes))
            self.trainable_parameters = model.fc.parameters()

        elif self.config.model_type == ModelType.EFFICIENTNET_B0:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
            self.trainable_parameters = model.classifier.parameters()

        elif self.config.model_type == ModelType.EFFICIENTNET_B1:
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
            self.trainable_parameters = model.classifier.parameters()

        elif self.config.model_type == ModelType.EFFICIENTNET_B2:
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
            self.trainable_parameters = model.classifier.parameters()

        elif self.config.model_type == ModelType.EFFICIENTNET_B3:
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes))
            self.trainable_parameters = model.classifier.parameters()

        else:
            raise ValueError(f"Modelo não suportado: {self.config.model_type}")

        return model.to(self.device)
    
    def _require_dataset(self) -> None:
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError("Esta operação requer datasets carregados.")

    def train_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float]:
        self._require_dataset()
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)

        for batch_index, (images, labels) in enumerate(self.train_loader, start=1):
            if self.cancel_requested:
                break

            if self.batch_progress_callback is not None:
                self.batch_progress_callback(epoch, total_epochs, batch_index, total_batches)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if total == 0:
            return 0.0, 0.0

        return total_loss / total, correct / total

    def validate(self) -> tuple[float, float]:
        self._require_dataset()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.cancel_requested:
                    break

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if total == 0:
            return 0.0, 0.0

        return total_loss / total, correct / total

    def train(self) -> dict[str, list[float]]:
        self._require_dataset()
        start_time = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            if self.cancel_requested:
                LOGGER.warning("Treinamento cancelado na época %d.", epoch)
                break

            train_loss, train_acc = self.train_epoch(epoch, self.config.epochs)
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if self.progress_callback is not None:
                self.progress_callback(
                    epoch,
                    self.config.epochs,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                )

            LOGGER.info(
                "Época %d/%d | Treino loss %.4f | Treino acc %.4f | Val loss %.4f | Val acc %.4f",
                epoch,
                self.config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )
        
        total_time = time.perf_counter() - start_time
        LOGGER.info("Treinamento concluído em %.2f segundos.", total_time)

        return self.history, total_time

    def save_model(self, path: Path | None = None) -> None:
        path = path or self.config.model_path
        checkpoint = {
            "model_type": self.config.model_type.value,
            "binary_classification": self.config.binary_classification,
            "dropout_rate": self.config.dropout_rate,
            "state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, path)
        LOGGER.info("Modelo salvo em %s", path)

    def load_model(self, path: Path | None = None) -> None:
        path = path or self.config.model_path
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        LOGGER.info("Modelo carregado de %s", path)

    def predict(self, image_tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            probabilities = torch.softmax(outputs, dim=1)[0].cpu()
            prediction = int(probabilities.argmax(dim=0).item())

        return prediction, probabilities
    
    def evaluate(self):
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.test_loader:

                images = images.to(self.device)

                outputs = self.model(images)

                predictions = outputs.argmax(dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(predictions.cpu().numpy())

        if self.config.binary_classification:
            return MetricsCalculator.binary_metrics(y_true,y_pred)

        return MetricsCalculator.multiclass_metrics(y_true,y_pred)
    
    @staticmethod
    def load_metadata(path: Path, device: torch.device) -> TrainingConfig:
        checkpoint = torch.load(path, map_location=device)
        return TrainingConfig(
            model_type=ModelType(checkpoint["model_type"]),
            binary_classification=checkpoint["binary_classification"],
            dropout_rate=checkpoint["dropout_rate"],
            model_path=path
        )

# =============================================================
# GRAD-CAM
# =============================================================

class GradCAMProcessor:
    def __init__(self, model: nn.Module):
        target_layer = self.get_target_layer(model)
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    @staticmethod
    def get_target_layer(model: nn.Module) -> nn.Module:
        if isinstance(model, models.ResNet):
            return model.layer4[-1]

        if isinstance(model, models.EfficientNet):
            return model.features[-1]

        raise ValueError(f"Modelo não suportado: {type(model)}")

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.enable_grad():
            grayscale_cam = self.cam(input_tensor=image_tensor)
        return grayscale_cam[0] if grayscale_cam is not None and len(grayscale_cam) > 0 else np.array([])

    def overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        visualization = show_cam_on_image(image, heatmap, use_rgb=True)
        return visualization

# =============================================================
# SERVICE CENTRALIZADO
# =============================================================

@dataclass
class ClassificationResult:
    prediction: int
    probabilities: torch.Tensor
    class_labels: list[str]
    is_binary: bool

@dataclass
class SegmentationResult:
    mask: np.ndarray
    segmented_image: np.ndarray

@dataclass
class GradCAMResult:
    visualization: np.ndarray | None
    error: str | None = None

class ApplicationService:
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.training_manager: TrainingManager | None = None
        self.gradcam_processor: GradCAMProcessor | None = None
        self.current_image: np.ndarray | None = None
        self.current_sample: Sample | None = None

    def load_dataset(self, dataset_path: str | Path) -> dict[str, int]:
        dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        self.dataset_manager.load_dataset(dataset_path)
        
        return {
            "total_samples": len(self.dataset_manager.samples),
            "train_samples": len(self.dataset_manager.train_samples),
            "test_samples": len(self.dataset_manager.test_samples),
        }

    def has_dataset(self) -> bool:
        return len(self.dataset_manager.samples) > 0

    def start_training(
        self,
        model_type: str,
        task_type: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        dropout_rate: float,
        use_segmentation: bool,
        segmentation_config: SegmentationConfig | None = None,
        progress_callback: Callable[[int, int, float, float, float, float], None] | None = None,
        batch_progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> bool:
        if not self.has_dataset():
            LOGGER.error("Nenhum dataset carregado")
            return False

        dataset_config = DatasetConfig(
            segmented=use_segmentation,
            augmentation=True,
            binary_classification=task_type == "binary",
            image_size=224,
            segmentation_config=segmentation_config or SegmentationConfig()
        )

        test_config = DatasetConfig(
            segmented=use_segmentation,
            augmentation=False,
            binary_classification=task_type == "binary",
            image_size=224,
            segmentation_config=segmentation_config or SegmentationConfig()
        )

        train_dataset = MammographyDataset(self.dataset_manager.train_samples, dataset_config)
        test_dataset = MammographyDataset(self.dataset_manager.test_samples, test_config)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            LOGGER.error("Dataset de treino ou teste vazio")
            return False

        model_folder = Path("models")
        model_folder.mkdir(parents=True, exist_ok=True)
        model_path = model_folder / f"{model_type}_{task_type}.pth"

        training_config = TrainingConfig(
            model_type=ModelType(model_type),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            binary_classification=task_type == "binary",
            model_path=model_path
        )

        self.training_manager = TrainingManager(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=training_config,
            progress_callback=progress_callback,
            batch_progress_callback=batch_progress_callback
        )

        return True

    def run_training(self) -> tuple[dict[str, list[float]], float, dict[str, Any] | None]:
        if not self.training_manager:
            raise RuntimeError("TrainingManager não foi inicializado. Chame start_training primeiro.")

        history, total_time = self.training_manager.train()
        metrics = None

        if not self.training_manager.cancel_requested:
            try:
                metrics = self.training_manager.evaluate()
            except Exception as e:
                LOGGER.error("Erro ao calcular métricas finais: %s", e)

        return history, total_time, metrics

    def cancel_training(self) -> None:
        if self.training_manager:
            self.training_manager.cancel_requested = True

    def save_model(self, path: Path | None = None) -> bool:
        if not self.training_manager:
            LOGGER.error("Nenhum modelo para salvar")
            return False

        try:
            self.training_manager.save_model(path)
            return True
        except Exception as e:
            LOGGER.error("Erro ao salvar modelo: %s", e)
            return False

    def load_model(self, path: Path) -> bool:
        training_config = TrainingManager.load_metadata(path, torch.device("cpu"))

        self.training_manager = TrainingManager(config=training_config)

        try:
            self.training_manager.load_model(path)
            self.gradcam_processor = GradCAMProcessor(self.training_manager.model)
            return True
        except Exception as e:
            LOGGER.error("Erro ao carregar modelo: %s", e)
            return False

    def load_image(self, image_path: str | Path) -> bool:
        try:
            path = Path(image_path)
            sample = Sample(
                path=path,
                filename=path.name,
                birads_class=SampleBiradsClass.D,
                number=1,
                set_type=SetType.TEST,
                side=SampleSide.RIGHT,
                view=SampleView.CC
            )
            self.current_sample = sample
            self.current_image = ImageManager.load(sample)
            return True
        except Exception as e:
            LOGGER.error("Erro ao carregar imagem: %s", e)
            return False

    def get_current_image(self) -> np.ndarray | None:
        return self.current_image

    def segment_image(self, image: np.ndarray | None = None, config: SegmentationConfig | None = None) -> SegmentationResult:
        if image is None:
            image = self.current_image

        if image is None:
            raise ValueError("Nenhuma imagem carregada")

        config = config or SegmentationConfig()
        mask, segmented = SegmentationProcessor.segment(image, config)
        
        return SegmentationResult(mask=mask, segmented_image=segmented)

    def _prepare_image_tensor(self, image: np.ndarray, use_segmentation: bool, segmentation_config: SegmentationConfig | None = None) -> tuple[torch.Tensor, np.ndarray]:
        config = segmentation_config or SegmentationConfig()
        
        processed = image.copy()
        if use_segmentation:
            _, processed = SegmentationProcessor.segment(processed, config)
        
        processed = ImageManager.resize(processed, 224, 224)
        
        processed = ImageManager.normalize(processed)
        
        processed_rgb = np.stack([processed] * 3, axis=-1)
        
        tensor = torch.from_numpy(processed_rgb.transpose((2, 0, 1))).unsqueeze(0).float()
        
        return tensor, processed_rgb

    def classify_image(self, image: np.ndarray, use_segmentation: bool, segmentation_config: SegmentationConfig | None = None) -> ClassificationResult:
        if not self.training_manager:
            raise RuntimeError("Nenhum modelo carregado para classificação")

        image_tensor, _ = self._prepare_image_tensor(image, use_segmentation, segmentation_config)

        prediction, probabilities = self.training_manager.predict(image_tensor)

        class_labels = list(DEFAULT_BIRADS_LABELS)

        return ClassificationResult(
            prediction=prediction,
            probabilities=probabilities,
            class_labels=class_labels,
            is_binary=self.training_manager.config.binary_classification
        )

    def generate_gradcam(
        self,
        image: np.ndarray,
        # task_type: str,
        use_segmentation: bool,
        segmentation_config: SegmentationConfig | None = None
    ) -> GradCAMResult:
        if not self.training_manager:
            return GradCAMResult(visualization=None, error="Nenhum modelo carregado")

        try:
            image_tensor, processed_rgb = self._prepare_image_tensor(image, use_segmentation, segmentation_config)

            if self.gradcam_processor is None:
                self.gradcam_processor = GradCAMProcessor(self.training_manager.model)

            self.training_manager.model.eval()
            gradcam_tensor = image_tensor.to(self.training_manager.device).requires_grad_(True)
            heatmap = self.gradcam_processor.generate(gradcam_tensor)
            
            if heatmap is None or heatmap.size == 0:
                return GradCAMResult(visualization=None, error="Heatmap vazio")
            
            overlay = self.gradcam_processor.overlay(processed_rgb, heatmap)

            return GradCAMResult(visualization=overlay)
        except Exception as e:
            LOGGER.error("Erro ao gerar Grad-CAM: %s", e, exc_info=True)
            return GradCAMResult(visualization=None, error=str(e))

# =============================================================
# INTERFACE GRÁFICA
# =============================================================

class TkinterLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        msg = self.format(record)
        self.callback(msg)

class GUI(tk.Tk):
    BIRADS_LABELS = DEFAULT_BIRADS_LABELS
    BINARY_RESULT_LABELS = ("BIRADS I/II", "BIRADS III/IV")
    BINARY_BAR_CLASS_INDEX = (0, 0, 1, 1)
    CLASSIFICATION_TASK_OPTIONS = ("Binária", "4 classes")
    CLASSIFICATION_TASK_VALUE_MAP = {
        "Binária": "binary",
        "4 classes": "four",
    }
    LEARNING_RATE_OPTIONS = ("0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10")
    CLASSIFICATION_PANELS = ("original", "mask", "segmented", "gradcam")
    PANEL_TITLES = {
        "original": "Imagem original",
        "mask": "Mascara",
        "segmented": "Imagem segmentada",
        "gradcam": "Grad-CAM",
    }
    
    def __init__(self) -> None:
        super().__init__()
        self.title("PAI - Segmentação e Classificação Mamográfica")
        self.app_service = ApplicationService()
        self.dataset_dir: Path | None = None
        self.training_thread: threading.Thread | None = None
        self.classification_images: dict[str, Image.Image] = {}
        self.classification_tk_images: dict[str, ImageTk.PhotoImage] = {}
        self.image_canvases: dict[str, tk.Canvas] = {}
        self.result_bars: dict[str, ttk.Progressbar] = {}
        self.result_percent_labels: dict[str, ttk.Label] = {}
        self.training_figures: dict[str, Figure] = {}
        self.training_axes: dict[str, Any] = {}
        self.training_plot_canvases: dict[str, FigureCanvasTkAgg] = {}
        self.confusion_figure: Figure | None = None
        self.confusion_axis: Any | None = None
        self.confusion_canvas: FigureCanvasTkAgg | None = None
        self.training_metric_labels: dict[str, ttk.Label] = {}
        self.training_metrics_status_label: ttk.Label | None = None
        self.dataset_status_label: ttk.Label | None = None
        self.classification_model_status_label: ttk.Label | None = None
        self.message_queue: queue.Queue[str] = queue.Queue()
        self.training_cancel_requested = False
        self.training_dataset_loaded = False
        self.training_is_running = False
        self.training_config_widgets: list[tk.Widget] = []
        self.training_segmentation_widgets: list[tk.Widget] = []
        self.training_config_summary_labels: dict[str, ttk.Label] = {}
        self.classification_task_display_label: ttk.Label | None = None
        self.classification_segmented_display_label: ttk.Label | None = None

        self.threshold_offset_var = tk.IntVar(value=_DEFAULT_SEGMENTATION_CONFIG.threshold_offset)
        self.closing_iterations_var = tk.IntVar(value=_DEFAULT_SEGMENTATION_CONFIG.closing_iterations)
        self.kernel_size_var = tk.IntVar(value=_DEFAULT_SEGMENTATION_CONFIG.kernel_size)
        self.crop_var = tk.BooleanVar(value=_DEFAULT_SEGMENTATION_CONFIG.crop)

        self.model_var = tk.StringVar(value=str(_DEFAULT_TRAINING_CONFIG.model_type))
        self.task_var = tk.StringVar(value="binary")
        self.classification_task_var = tk.StringVar(value="Binária")
        self.binary_classification_var = tk.BooleanVar(value=_DEFAULT_TRAINING_CONFIG.binary_classification)
        self.segmented_var = tk.BooleanVar(value=True)
        self.classification_show_segmented_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.epochs_var = tk.IntVar(value=_DEFAULT_TRAINING_CONFIG.epochs)
        self.batch_var = tk.IntVar(value=_DEFAULT_TRAINING_CONFIG.batch_size)
        self.dropout_var = tk.DoubleVar(value=_DEFAULT_TRAINING_CONFIG.dropout_rate)
        self.lr_var = tk.StringVar(value=str(_DEFAULT_TRAINING_CONFIG.learning_rate))

        self.build_layout()
        self.set_training_buttons_state(True)
        self._setup_logging_handler()

        self.model_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.binary_classification_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.epochs_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.batch_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.lr_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.dropout_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.segmented_var.trace_add("write", lambda *_: self.update_training_config_summary())
        self.after(200, self.flush_messages)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def build_layout(self) -> None:
        self.minsize(1180, 700)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar_frame = ttk.Frame(self, padding=10, width=240)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns")
        self.sidebar_frame.grid_propagate(False)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.training_page = ttk.Frame(self.notebook, padding=10)
        self.classification_page = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.training_page, text="Treinamento")
        self.notebook.add(self.classification_page, text="Classificação")

        self.build_training_page()
        self.build_classification_page()
        self.build_sidebar_for_training()

    def build_training_page(self) -> None:
        self.training_page.columnconfigure(0, weight=1)
        self.training_page.rowconfigure(0, weight=2)
        self.training_page.rowconfigure(1, weight=2)
        self.training_page.rowconfigure(2, weight=1)

        graph_area = ttk.Frame(self.training_page)
        graph_area.grid(row=0, column=0, sticky="nsew")
        graph_area.columnconfigure(0, weight=1)
        graph_area.columnconfigure(1, weight=1)
        graph_area.rowconfigure(0, weight=1)

        chart_specs = (
            ("loss", "Loss", "Loss"),
            ("accuracy", "Acurácia", "Acurácia"),
        )
        for column, (chart_key, title, ylabel) in enumerate(chart_specs):
            self.create_chart_panel(graph_area, chart_key, column, title, ylabel)

        analytics_area = ttk.Frame(self.training_page)
        analytics_area.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        analytics_area.columnconfigure(0, weight=3)
        analytics_area.columnconfigure(1, weight=2)
        analytics_area.rowconfigure(0, weight=1)

        confusion_frame = ttk.LabelFrame(analytics_area, text="Matriz de confusão", padding=8)
        confusion_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        confusion_frame.rowconfigure(0, weight=1)
        confusion_frame.columnconfigure(0, weight=1)
        self.confusion_figure = Figure(figsize=(4.8, 2.8), dpi=100)
        self.confusion_axis = self.confusion_figure.add_subplot(111)
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_figure, master=confusion_frame)
        self.confusion_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.render_confusion_matrix(None)

        metrics_frame = ttk.LabelFrame(analytics_area, text="Métricas finais", padding=10)
        metrics_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        metrics_frame.columnconfigure(1, weight=1)
        self.training_metrics_status_label = ttk.Label(metrics_frame, text="Aguardando avaliação.")
        self.training_metrics_status_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        metric_rows = (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
            ("f1", "F1"),
            ("mean_sensitivity", "Sensibilidade média"),
            ("mean_specificity", "Especificidade média"),
        )
        for row, (key, label) in enumerate(metric_rows, start=1):
            ttk.Label(metrics_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            value_label = ttk.Label(metrics_frame, text="-", anchor="e")
            value_label.grid(row=row, column=1, sticky="e", pady=2)
            self.training_metric_labels[key] = value_label

        log_frame = ttk.LabelFrame(self.training_page, text="Logs", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def build_classification_page(self) -> None:
        self.classification_page.columnconfigure(0, weight=1)
        self.classification_page.rowconfigure(0, weight=3)
        self.classification_page.rowconfigure(1, weight=2)

        image_area = ttk.Frame(self.classification_page)
        image_area.grid(row=0, column=0, sticky="nsew")
        image_area.rowconfigure(0, weight=1)
        
        for column in range(len(self.CLASSIFICATION_PANELS)):
            image_area.columnconfigure(column, weight=1, uniform="classification_images")
        
        for column, key in enumerate(self.CLASSIFICATION_PANELS):
            self.create_image_panel_with_canvas(image_area, key, column, len(self.CLASSIFICATION_PANELS))

        result_frame = ttk.LabelFrame(self.classification_page, text="Resultado final da classificação", padding=10)
        result_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, height=4, wrap="word")
        self.result_text.grid(row=0, column=0, sticky="ew")
        self.result_text.insert("end", "Nenhuma classificação executada.")
        self.result_text.configure(state="disabled")

        metrics_frame = ttk.Frame(result_frame)
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        metrics_frame.columnconfigure(1, weight=1)
        for row, label in enumerate(self.BIRADS_LABELS):
            ttk.Label(metrics_frame, text=label, width=12).grid(row=row, column=0, sticky="w", pady=2)
            bar = ttk.Progressbar(metrics_frame, maximum=100, value=0)
            bar.grid(row=row, column=1, sticky="ew", padx=8, pady=2)
            percent_label = ttk.Label(metrics_frame, text="0%", width=6)
            percent_label.grid(row=row, column=2, sticky="e", pady=2)
            self.result_bars[label] = bar
            self.result_percent_labels[label] = percent_label

    @staticmethod
    def panel_padding(column: int, total_columns: int) -> tuple[int, int]:
        if column == 0:
            return (0, 6)
        if column == total_columns - 1:
            return (6, 0)
        return (6, 6)

    def create_image_panel_with_canvas(self, parent: ttk.Frame, key: str, column: int, total_columns: int) -> tk.Canvas:
        frame = ttk.LabelFrame(parent, text=self.PANEL_TITLES[key], padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=self.panel_padding(column, total_columns))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, background="#111111", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        canvas.bind("<Configure>", lambda _: self.render_classification_panel(key))
        self.image_canvases[key] = canvas
        return canvas

    def create_chart_panel(self, parent: ttk.Frame, key: str, column: int, title: str, ylabel: str) -> Figure:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        figure = Figure(figsize=(4.2, 2.6), dpi=100)
        axis = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.training_figures[key] = figure
        self.training_axes[key] = axis
        self.training_plot_canvases[key] = canvas
        self._draw_training_history_chart(key, [], [], title, ylabel)
        return figure

    def add_control_section(self, parent: ttk.Frame, row: int, title: str | None = None) -> int:
        if title is not None:
            ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", pady=(10, 0) if row > 0 else 0)
            return row + 1
        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=8)
        return row + 1

    def add_control(
        self,
        parent: ttk.Frame,
        row: int,
        label_text: str,
        control_widget: tk.Widget,
        pady_label: tuple[int, int] = (0, 0),
        pady_control: tuple[int, int] = (0, 0),
    ) -> int:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=pady_label)
        control_widget.grid(row=row + 1, column=0, sticky="ew", pady=pady_control)
        return row + 2

    def on_tab_changed(self, _event: tk.Event | None = None) -> None:
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        if tab_text == "Classificação":
            self.build_sidebar_for_classification()
        else:
            self.build_sidebar_for_training()

    def clear_sidebar(self) -> ttk.Frame:
        for child in self.sidebar_frame.winfo_children():
            child.destroy()
        self.sidebar_frame.columnconfigure(0, weight=1)
        return self.sidebar_frame

    def build_sidebar_for_training(self) -> None:
        panel = self.clear_sidebar()
        self.training_config_widgets = []
        self.training_segmentation_widgets = []

        row = 0
        ttk.Label(panel, text="Treinamento", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Button(panel, text="Selecionar dataset", command=self.choose_dataset).grid(
            row=row,
            column=0,
            sticky="ew",
            pady=(12, 3)
        )
        row += 1

        dataset_text = "Conjunto de dados não selecionado"
        if self.dataset_dir is not None:
            dataset_text = f"Conjunto de dados: {self.dataset_dir.name}"
        self.dataset_status_label = ttk.Label(panel, text=dataset_text, wraplength=220, justify="left")
        self.dataset_status_label.grid(row=row, column=0, sticky="w")
        row += 1

        row = self.add_control_section(panel, row)
        ttk.Label(panel, text="Configurações de treino").grid(row=row, column=0, sticky="w")
        row += 1

        model_combo = ttk.Combobox(
            panel,
            textvariable=self.model_var,
            values=["resnet18", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"],
            state="readonly"
        )
        row = self.add_control(panel, row, "Modelo", model_combo, pady_label=(8, 0))
        self.training_config_widgets.append(model_combo)

        batch_spin = ttk.Spinbox(panel, from_=1, to=256, textvariable=self.batch_var, width=8)
        row = self.add_control(panel, row, "Tamanho do batch", batch_spin, pady_label=(8, 0))
        self.training_config_widgets.append(batch_spin)

        lr_combo = ttk.Combobox(
            panel,
            textvariable=self.lr_var,
            values=list(self.LEARNING_RATE_OPTIONS),
            state="readonly"
        )
        row = self.add_control(panel, row, "Taxa de aprendizado", lr_combo, pady_label=(8, 0))
        self.training_config_widgets.append(lr_combo)

        binary_check = ttk.Checkbutton(panel, text="Classificação binária", variable=self.binary_classification_var)
        binary_check.grid(row=row, column=0, sticky="w", pady=(8, 2))
        self.training_config_widgets.append(binary_check)
        row += 1

        epochs_spin = ttk.Spinbox(panel, from_=1, to=200, textvariable=self.epochs_var, width=8)
        row = self.add_control(panel, row, "Número de épocas", epochs_spin, pady_label=(8, 0))
        self.training_config_widgets.append(epochs_spin)

        dropout_spin = ttk.Spinbox(panel, from_=0.0, to=10.0, increment=0.05, textvariable=self.dropout_var, width=8)
        row = self.add_control(panel, row, "Taxa de dropout", dropout_spin, pady_label=(8, 0))
        self.training_config_widgets.append(dropout_spin)

        segmented_check = ttk.Checkbutton(
            panel,
            text="Usar imagens segmentadas",
            variable=self.segmented_var,
            command=self.update_training_segmentation_controls_state,
        )
        segmented_check.grid(row=row, column=0, sticky="w", pady=(8, 2))
        self.training_config_widgets.append(segmented_check)
        row += 1

        ttk.Label(panel, text="Configuração de segmentação").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1

        threshold_spin = ttk.Spinbox(panel, from_=-100, to=100, textvariable=self.threshold_offset_var)
        row = self.add_control(panel, row, "Ajuste do limiar", threshold_spin)
        self.training_segmentation_widgets.append(threshold_spin)

        closing_spin = ttk.Spinbox(panel, from_=1, to=100, textvariable=self.closing_iterations_var)
        row = self.add_control(panel, row, "Iterações de fechamento", closing_spin)
        self.training_segmentation_widgets.append(closing_spin)

        kernel_spin = ttk.Spinbox(panel, from_=10, to=150, increment=1, textvariable=self.kernel_size_var)
        row = self.add_control(panel, row, "Tamanho do kernel", kernel_spin)
        self.training_segmentation_widgets.append(kernel_spin)

        crop_check = ttk.Checkbutton(panel, text="Recortar região de interesse (ROI)", variable=self.crop_var)
        crop_check.grid(row=row, column=0, sticky="w", pady=(2, 6))
        self.training_segmentation_widgets.append(crop_check)
        row += 1

        row = self.add_control_section(panel, row)

        self.start_training_button = ttk.Button(panel, text="Treinar dataset/modelo", command=self.start_training)
        self.start_training_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1
        
        self.cancel_training_button = ttk.Button(panel, text="Cancelar treinamento", command=self.cancel_training)
        self.cancel_training_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1
        
        self.export_model_button = ttk.Button(panel, text="Exportar modelo treinado", command=self.export_trained_model)
        self.export_model_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        ttk.Label(panel, text="Progresso de treino").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        
        self.training_progress_bar = ttk.Progressbar(panel, maximum=100)
        self.training_progress_bar.grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        
        self.training_status_label = ttk.Label(panel, text="Aguardando treino...")
        self.training_status_label.grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Separator(panel).grid(row=row, column=0, sticky="ew", pady=(10, 4))
        row += 1
        ttk.Button(panel, text="Limpar logs", command=self._clear_logs).grid(row=row, column=0, sticky="ew", pady=3)

        self.update_training_controls_state()
        self.update_training_config_summary()

    def build_sidebar_for_classification(self) -> None:
        panel = self.clear_sidebar()
        row = 0
        ttk.Label(panel, text="Classificação", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1

        self.classification_model_status_label = ttk.Label(panel, text="Modelo: não carregado", wraplength=220, justify="left")
        self.classification_model_status_label.grid(row=row, column=0, sticky="w", pady=(8, 4))
        row += 1

        self.classification_import_button = ttk.Button(panel, text="Importar modelo", command=self.load_existing_model)
        self.classification_import_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        self.classification_open_image_button = ttk.Button(panel, text="Selecionar imagem", command=self.open_image)
        self.classification_open_image_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        row = self.add_control_section(panel, row)

        ttk.Label(panel, text="Configuração do Modelo Importado").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1

        ttk.Label(panel, text="Tipo de classificação:").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        self.classification_task_display_label = ttk.Label(panel, text="-")
        self.classification_task_display_label.grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        ttk.Label(panel, text="Treinado com imagens segmentadas:").grid(row=row, column=0, sticky="w")
        row += 1
        self.classification_segmented_display_label = ttk.Label(panel, text="-")
        self.classification_segmented_display_label.grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        self.classify_button = ttk.Button(panel, text="Classificar imagem", command=self.classify_current_image)
        self.classify_button.grid(row=row, column=0, sticky="ew", pady=3)
        row += 1

        row = self.add_control_section(panel, row)
        ttk.Label(panel, text="Zoom").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Scale(panel, from_=0.2, to=3.0, variable=self.zoom_var, command=lambda _value: self.refresh_image()).grid(row=row, column=0, sticky="ew")
        row += 1
        ttk.Button(panel, text="Zoom 100%", command=self.reset_zoom).grid(row=row, column=0, sticky="ew", pady=3)

        self.update_classification_controls_state()

    def update_classification_controls_state(self) -> None:
        has_model = self.app_service.training_manager is not None
        has_image = self.app_service.get_current_image() is not None

        if self.classification_model_status_label is not None:
            self.classification_model_status_label.config(
                text="Modelo: " + self.app_service.training_manager.config.model_type 
                if has_model 
                else "Modelo: não carregado (importe para classificar)"
            )

        if hasattr(self, "classification_import_button"):
            self.classification_import_button.config(state="normal")
        if hasattr(self, "classification_open_image_button"):
            self.classification_open_image_button.config(state="normal" if has_model else "disabled")
        if hasattr(self, "classify_button"):
            self.classify_button.config(state="normal" if (has_model and has_image) else "disabled")

        if (has_model and self.app_service.training_manager and 
            self.classification_task_display_label is not None and 
            self.classification_segmented_display_label is not None):
            task_label = "Binária" if self.app_service.training_manager.config.binary_classification else "4 classes"
            self.classification_task_display_label.config(text=task_label)
            self.classification_segmented_display_label.config(
                text="Sim" if self.classification_show_segmented_var.get() else "Não"
            )
        elif (self.classification_task_display_label is not None and 
              self.classification_segmented_display_label is not None):
            self.classification_task_display_label.config(text="-")
            self.classification_segmented_display_label.config(text="-")

    def _update_classification_preview(self) -> None:
        current_image = self.app_service.get_current_image()
        if current_image is None:
            return

        image_pil = ImageManager.to_pil(current_image)
        self.set_classification_image("original", image_pil)

        if self.classification_show_segmented_var.get():
            seg_result = self.app_service.segment_image(config=self.get_segmentation_config())
            self.set_classification_image("mask", ImageManager.to_pil(seg_result.mask * 255))
            self.set_classification_image("segmented", ImageManager.to_pil(seg_result.segmented_image * 255))
        else:
            self.set_classification_image("mask", None)
            self.set_classification_image("segmented", None)

        self.set_classification_image("gradcam", None)

    # def _get_classification_task_type(self) -> str:
    #     return self.CLASSIFICATION_TASK_VALUE_MAP.get(self.classification_task_var.get(), "binary")

    def set_classification_image(self, key: str, image: Image.Image | None) -> None:
        self.classification_images[key] = image
        self.render_classification_panel(key)

    def render_classification_panel(self, key: str) -> None:
        canvas = self.image_canvases.get(key)
        if canvas is None:
            return

        image = self.classification_images.get(key)
        if image is None:
            self.draw_placeholder(canvas, self.PANEL_TITLES.get(key, key))
            return

        canvas_w = max(canvas.winfo_width(), 1)
        canvas_h = max(canvas.winfo_height(), 1)
        zoom = self.zoom_var.get()
        target_w = max(1, int(canvas_w * zoom))
        target_h = max(1, int(canvas_h * zoom))

        resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized)
        self.classification_tk_images[key] = tk_image

        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, anchor="center", image=tk_image)

    def reset_zoom(self) -> None:
        self.zoom_var.set(1.0)
        self.refresh_image()

    @staticmethod
    def draw_placeholder(canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_text(width / 2, height / 2, text=text, fill="#777777", font=("TkDefaultFont", 14))

    def log_message(self, message: str) -> None:
        self.message_queue.put(message)

    def flush_messages(self) -> None:
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.log.insert("end", message + "\n")
            self.log.see("end")
        self.after(200, self.flush_messages)

    def load_dataset(self, dataset_dir: Path) -> None:
        try:
            self.dataset_dir = dataset_dir
            stats = self.app_service.load_dataset(dataset_dir)
            self.training_dataset_loaded = stats["total_samples"] > 0
            if self.dataset_status_label is not None:
                self.dataset_status_label.config(
                    text=f"Conjunto de dados: {dataset_dir.name} ({stats['train_samples']} treino / {stats['test_samples']} teste)"
                )
        finally:
            self.update_training_controls_state()

    def choose_dataset(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.dataset_dir if self.dataset_dir and self.dataset_dir.exists() else Path.cwd()))
        if directory:
            self.load_dataset(Path(directory))

    def get_segmentation_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            threshold_offset=self.threshold_offset_var.get(),
            closing_iterations=self.closing_iterations_var.get(),
            kernel_size=self.kernel_size_var.get(),
            crop=self.crop_var.get()
        )

    def start_training(self) -> None:
        if not self.app_service.has_dataset():
            messagebox.showinfo("Dataset", "Selecione um dataset primeiro.")
            return

        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Treinamento", "Um treinamento já está em execução.")
            return

        try:
            task_type = "binary" if self.binary_classification_var.get() else "four"
            self.task_var.set(task_type)

            success = self.app_service.start_training(
                model_type=self.model_var.get(),
                task_type=task_type,
                epochs=self.epochs_var.get(),
                batch_size=self.batch_var.get(),
                learning_rate=float(self.lr_var.get()),
                dropout_rate=float(self.dropout_var.get()),
                use_segmentation=self.segmented_var.get(),
                segmentation_config=self.get_segmentation_config(),
                progress_callback=self.update_training_progress,
                batch_progress_callback=self.update_training_batch_progress,
            )

            if not success:
                messagebox.showinfo("Treinamento", "O dataset precisa conter amostras de treino e teste. Verifique o conjunto carregado.")
                return

            self.training_progress_bar["value"] = 0
            self.training_status_label.config(text="Preparando treino...")
            self.clear_training_outputs()
            self.training_is_running = True
            self.set_training_buttons_state(False)
            LOGGER.info("Iniciando treinamento em segundo plano...")

            self.training_thread = threading.Thread(target=self._run_training, daemon=True)
            self.training_thread.start()

        except Exception as exc:
            LOGGER.error("Erro ao iniciar treinamento: %s", exc)
            messagebox.showerror("Erro", str(exc))
            self.training_is_running = False
            self.set_training_buttons_state(True)

    def _run_training(self) -> None:
        try:
            history, total_time, metrics = self.app_service.run_training()
            LOGGER.info("Treinamento finalizado em %.2fs. Use 'Exportar modelo' para salvar.", total_time)
            self.after(0, self.render_training_history)
            self.after(0, lambda: self.render_training_evaluation(metrics))
            if metrics and metrics.get("report"):
                LOGGER.info("Relatório de classificação:\n%s", metrics["report"])
        except Exception as exc:
            LOGGER.error("Erro durante treinamento: %s", exc)
            self.after(0, lambda: messagebox.showerror("Erro", str(exc)))
        finally:
            self.training_is_running = False
            self.after(0, lambda: self.set_training_buttons_state(True))

    def cancel_training(self) -> None:
        self.app_service.cancel_training()
        LOGGER.info("Solicitação de cancelamento enviada. O treinamento será interrompido após a etapa atual.")

    def export_trained_model(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth")],
            initialfile="model.pth",
            initialdir=str(Path("models").resolve())
        )

        if not path:
            return

        try:
            self.app_service.save_model(Path(path))
            messagebox.showinfo("Exportar modelo", f"Modelo exportado para {path}")
        except Exception as exc:
            LOGGER.error("Erro ao exportar modelo: %s", exc)
            messagebox.showerror("Erro", str(exc))

    def set_training_buttons_state(self, enabled: bool) -> None:
        self.training_is_running = not enabled
        self.update_training_controls_state()

    @staticmethod
    def _set_widget_enabled(widget: tk.Widget, enabled: bool) -> None:
        try:
            if isinstance(widget, ttk.Combobox):
                widget.config(state="readonly" if enabled else "disabled")
            else:
                widget.config(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass

    def update_training_controls_state(self) -> None:
        config_enabled = self.training_dataset_loaded and not self.training_is_running

        for widget in self.training_config_widgets:
            self._set_widget_enabled(widget, config_enabled)

        self.update_training_segmentation_controls_state()

        has_model = self.app_service.training_manager is not None
        if hasattr(self, "start_training_button"):
            self.start_training_button.config(state="normal" if config_enabled else "disabled")
        if hasattr(self, "export_model_button"):
            self.export_model_button.config(state="normal" if (has_model and not self.training_is_running) else "disabled")
        if hasattr(self, "cancel_training_button"):
            self.cancel_training_button.config(state="normal" if self.training_is_running else "disabled")

    def update_training_config_summary(self) -> None:
        if not self.training_config_summary_labels:
            return

        task_label = "Binária" if self.binary_classification_var.get() else "4 classes"
        for key, label_text in (
            ("model", self.model_var.get()),
            ("task", task_label),
            ("epochs", str(self.epochs_var.get())),
            ("batch", str(self.batch_var.get())),
            ("lr", self.lr_var.get()),
            ("dropout", str(self.dropout_var.get())),
            ("segmented", "Sim" if self.segmented_var.get() else "Não"),
        ):
            label_widget = self.training_config_summary_labels.get(key)
            if label_widget is not None:
                label_widget.config(text=label_text)

    def update_training_segmentation_controls_state(self) -> None:
        segmentation_enabled = (
            self.training_dataset_loaded
            and not self.training_is_running
            and self.segmented_var.get()
        )
        for widget in self.training_segmentation_widgets:
            self._set_widget_enabled(widget, segmentation_enabled)

    def update_training_batch_progress(self, epoch: int, total_epochs: int, batch_index: int, total_batches: int,) -> None:
        if total_batches <= 0 or total_epochs <= 0:
            return

        completed_steps = (epoch - 1) * total_batches + batch_index
        total_steps = total_epochs * total_batches
        percent = (completed_steps / total_steps) * 100.0

        self.after(0, lambda: self.training_progress_bar.config(value=percent))
        self.after(0, lambda: self.training_status_label.config(
                text=f"Época {epoch}/{total_epochs} | Lote {batch_index}/{total_batches} | {percent:.1f}%"))

    def update_training_progress(self, epoch: int, total_epochs: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
        self.after(
            0,
            lambda: self.training_status_label.config(
                text=(
                    f"Época {epoch}/{total_epochs} concluída | perda {train_loss:.4f} | "
                    f"acurácia {train_acc:.4f} | val {val_acc:.4f}"
                )
            ),
        )
        self.after(0, self.render_training_history)

    def render_training_history(self) -> None:
        training_manager = self.app_service.training_manager
        if training_manager is None:
            return

        history = training_manager.history
        self._draw_training_history_chart(
            "loss",
            history["train_loss"],
            history["val_loss"],
            "Loss",
            "Loss"
        )
        self._draw_training_history_chart(
            "accuracy",
            history["train_accuracy"],
            history["val_accuracy"],
            "Acurácia",
            "Acurácia"
        )

    def _draw_training_history_chart(self, chart_key: str, train_values: list[float], val_values: list[float], title: str, ylabel: str,) -> None:
        axis = self.training_axes.get(chart_key)
        figure = self.training_figures.get(chart_key)
        canvas = self.training_plot_canvases.get(chart_key)
        if axis is None or figure is None or canvas is None:
            return

        axis.clear()

        if not train_values:
            axis.set_axis_off()
            axis.text(0.5, 0.5, "Sem dados", ha="center", va="center", color="#777777")
            figure.tight_layout()
            canvas.draw_idle()
            return

        train_epochs = np.arange(1, len(train_values) + 1)
        axis.plot(train_epochs, train_values, marker="o", linewidth=2, label="Treino")

        if val_values:
            val_epochs = np.arange(1, len(val_values) + 1)
            axis.plot(val_epochs, val_values, marker="o", linewidth=2, label="Validação")

        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

        if chart_key == "accuracy":
            all_values = train_values + val_values
            upper_limit = max(1.0, max(all_values) * 1.05)
            axis.set_ylim(0, upper_limit)

        figure.tight_layout()
        canvas.draw_idle()

    def clear_training_outputs(self) -> None:
        self._draw_training_history_chart("loss", [], [], "Loss", "Loss")
        self._draw_training_history_chart("accuracy", [], [], "Acurácia", "Acurácia")
        self.render_training_evaluation(None)

    def render_training_evaluation(self, metrics: dict[str, Any] | None) -> None:
        self.render_training_metrics(metrics)
        self.render_confusion_matrix(metrics)

    def render_training_metrics(self, metrics: dict[str, Any] | None) -> None:
        for label in self.training_metric_labels.values():
            label.config(text="-")

        if self.training_metrics_status_label is None:
            return

        if not metrics:
            self.training_metrics_status_label.config(text="Aguardando avaliação.")
            return

        self.training_metrics_status_label.config(text="Avaliação do conjunto de teste")
        metric_values = {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "f1": metrics.get("f1"),
            "mean_sensitivity": metrics.get("mean_sensitivity"),
            "mean_specificity": metrics.get("mean_specificity"),
        }

        for key, value in metric_values.items():
            label = self.training_metric_labels.get(key)
            if label is not None and value is not None:
                label.config(text=self.format_metric_value(value))

    def render_confusion_matrix(self, metrics: dict[str, Any] | None) -> None:
        if self.confusion_axis is None or self.confusion_figure is None or self.confusion_canvas is None:
            return

        self.confusion_axis.clear()
        matrix = None if not metrics else metrics.get("confusion_matrix")
        if matrix is None:
            self.confusion_axis.set_axis_off()
            self.confusion_axis.text(0.5, 0.5, "Matriz indisponível", ha="center", va="center", color="#777777")
            self.confusion_figure.tight_layout()
            self.confusion_canvas.draw_idle()
            return

        matrix = np.asarray(matrix)
        labels = self.training_display_labels(matrix.shape[0])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        display.plot(ax=self.confusion_axis, cmap="Blues", colorbar=False, values_format="d")
        self.confusion_axis.set_title("Matriz de Confusão")
        self.confusion_axis.set_xlabel("Predito")
        self.confusion_axis.set_ylabel("Real")
        self.confusion_figure.tight_layout()
        self.confusion_canvas.draw_idle()

    def training_display_labels(self, class_count: int) -> list[str]:
        if class_count == 2:
            return ["I/II", "III/IV"]
        if class_count == 4:
            return ["I", "II", "III", "IV"]
        return [str(index) for index in range(class_count)]

    @staticmethod
    def format_metric_value(value: Any) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "-"

    def load_existing_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")], initialdir=str(Path("models").resolve()))

        if not path:
            return

        try:
            if self.app_service.load_model(Path(path)):
                LOGGER.info("Modelo carregado de %s", path)
                self.update_classification_controls_state()
            else:
                messagebox.showerror("Erro", "Falha ao carregar o modelo. Verifique se o dataset está carregado.")
        except Exception as exc:
            LOGGER.error("Erro ao carregar modelo: %s", exc)
            messagebox.showerror("Erro", str(exc))

    def update_gradcam_panel(self) -> None:
        if not self.app_service.training_manager or self.app_service.get_current_image() is None:
            self.set_classification_image("gradcam", None)
            return

        try:
            LOGGER.debug("Iniciando geração de Grad-CAM...")
            image = self.app_service.get_current_image()
            LOGGER.debug("Imagem atual shape: %s, dtype: %s", image.shape, image.dtype)
            
            gradcam_result = self.app_service.generate_gradcam(
                image,
                # task_type=self._get_classification_task_type(),
                use_segmentation=self.classification_show_segmented_var.get(),
                segmentation_config=self.get_segmentation_config()
            )
            
            if gradcam_result.visualization is not None:
                LOGGER.debug("Grad-CAM gerado com sucesso: shape %s", gradcam_result.visualization.shape)
                self.set_classification_image("gradcam", Image.fromarray(gradcam_result.visualization.astype(np.uint8)))
            else:
                self.set_classification_image("gradcam", None)
                if gradcam_result.error:
                    LOGGER.warning("Grad-CAM indisponível: %s", gradcam_result.error)
                else:
                    LOGGER.warning("Grad-CAM retornou visualização None")

        except Exception as exc:
            self.set_classification_image("gradcam", None)
            LOGGER.error("Erro ao gerar Grad-CAM: %s", exc, exc_info=True)

    def open_image(self) -> None:
        if self.app_service.training_manager is None:
            messagebox.showinfo("Classificação", "Importe um modelo antes de selecionar a imagem.")
            return

        filename = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.tif *.tiff")])

        if not filename:
            return

        try:
            if not self.app_service.load_image(filename):
                messagebox.showerror("Erro", f"Não foi possível abrir a imagem: {filename}")
                return

            self._update_classification_preview()

            self.zoom_var.set(1.0)
            self.refresh_image()
            self.update_classification_controls_state()

            LOGGER.info(f"Imagem aberta: {filename}")

        except Exception as exc:
            LOGGER.error(f"Erro ao abrir imagem: {exc}")
            messagebox.showerror("Erro", str(exc))

    def refresh_image(self) -> None:
        for name in self.CLASSIFICATION_PANELS:
            self.render_classification_panel(name)

    def _display_classification_result(self, result: ClassificationResult) -> str:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")

        if result.is_binary:
            label = self.BINARY_RESULT_LABELS[result.prediction]
            self.result_text.insert("end", f"Resultado binário: {label}\n")
            self.result_text.insert("end", f"Confiança: {result.probabilities[result.prediction].item() * 100:.2f}%\n")
            confidence_0 = result.probabilities[0].item() * 100
            confidence_1 = result.probabilities[1].item() * 100
            binary_confidences = (confidence_0, confidence_1)
            for label_name, class_index in zip(self.BIRADS_LABELS, self.BINARY_BAR_CLASS_INDEX):
                value = binary_confidences[class_index]
                self.result_bars[label_name].config(value=value)
                self.result_percent_labels[label_name].config(text=f"{value:.1f}%")
        else:
            label = result.class_labels[result.prediction]
            self.result_text.insert("end", f"Resultado: {label}\n")
            self.result_text.insert("end", "Probabilidades por classe:\n")
            for index, class_name in enumerate(result.class_labels):
                self.result_text.insert("end", f"  {class_name}: {result.probabilities[index].item() * 100:.2f}%\n")
            for class_name, prob_value in zip(result.class_labels, result.probabilities.tolist()):
                self.result_bars[class_name].config(value=prob_value * 100)
                self.result_percent_labels[class_name].config(text=f"{prob_value * 100:.1f}%")

        self.result_text.configure(state="disabled")
        return label

    def classify_current_image(self) -> None:
        if self.app_service.get_current_image() is None:
            messagebox.showinfo("Classificação", "Abra uma imagem para classificar.")
            return

        if not self.app_service.training_manager:
            messagebox.showinfo("Classificação", "Carregue um modelo ou treine um modelo primeiro.")
            return

        try:
            result = self.app_service.classify_image(
                self.app_service.get_current_image(),
                # task_type=self._get_classification_task_type(),
                use_segmentation=self.classification_show_segmented_var.get(),
                segmentation_config=self.get_segmentation_config()
            )
            label = self._display_classification_result(result)
            self.update_gradcam_panel()
            LOGGER.info("Classificação concluída: %s", label)
        except Exception as exc:
            LOGGER.exception("Erro ao classificar imagem: %s", exc)
            messagebox.showerror("Erro", str(exc))

    def _clear_logs(self) -> None:
        self.log.delete("1.0", "end")

    def _on_close(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            if not messagebox.askyesno("Sair", "Há um treinamento em andamento. Deseja encerrar mesmo assim?"):
                return
            self.app_service.cancel_training()

        if self.app_service.training_manager is not None:
            answer = messagebox.askyesnocancel("Sair", "Há um modelo em memória. Deseja salvar antes de sair?")
            if answer is None:
                return
            if answer:
                self.export_trained_model()

        self.destroy()

    def _setup_logging_handler(self) -> None:
        def log_callback(msg: str) -> None:
            self.log_message(msg)

        handler = TkinterLogHandler(log_callback)
        handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)

# =============================================================
# MAIN
# =============================================================

def main():
    app = GUI()
    app.mainloop()

if __name__ == "__main__":
    main()
