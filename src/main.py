"""
Trabalho Pratico - Processamento e Analise de Imagens
Segmentacao e classificacao de imagens mamograficas.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from PIL import Image, ImageOps, ImageTk
except Exception as exc:  # pragma: no cover - mensagem amigavel em runtime
    raise SystemExit("Instale Pillow para executar a aplicacao: pip install pillow") from exc

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise SystemExit("Tkinter nao esta disponivel neste Python.") from exc

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    models = None
    transforms = None


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}
CLASS_NAMES = ["D", "E", "F", "G"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
BIRADS_LABELS = {
    "D": "BIRADS I",
    "E": "BIRADS II",
    "F": "BIRADS III",
    "G": "BIRADS IV",
}
ROTATION_ANGLES = [-20, -10, 0, 10, 20]
DEFAULT_DATASET_DIR = Path("Dataset") / "RCC"
MODEL_DIR = Path("models")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str
    class_index: int
    number: int
    split: str

    @property
    def binary_index(self) -> int:
        return 0 if self.class_name in {"D", "E"} else 1


def natural_number_from_name(path: Path) -> int:
    """Extrai a numeracao usada na regra de treino/teste."""
    matches = re.findall(r"\((\d+)\)|(\d+)", path.stem)
    if not matches:
        return 1
    last = matches[-1]
    return int(last[0] or last[1])


def discover_dataset(dataset_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.strip()[0].upper()
        if class_name not in CLASS_TO_INDEX:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            number = natural_number_from_name(path)
            split = "test" if number % 4 == 0 else "train"
            records.append(
                ImageRecord(
                    path=path,
                    class_name=class_name,
                    class_index=CLASS_TO_INDEX[class_name],
                    number=number,
                    split=split,
                )
            )
    return records


def summarize_records(records: Iterable[ImageRecord]) -> str:
    records = list(records)
    lines = [f"Total: {len(records)} imagens"]
    for class_name in CLASS_NAMES:
        subset = [item for item in records if item.class_name == class_name]
        train = sum(item.split == "train" for item in subset)
        test = sum(item.split == "test" for item in subset)
        lines.append(f"{class_name} ({BIRADS_LABELS[class_name]}): {len(subset)} | treino={train} teste={test}")
    return "\n".join(lines)


def load_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in {"L", "I;16", "I", "F"}:
        image = image.convert("L")
    array = np.asarray(image).astype(np.float32)
    max_value = float(np.max(array)) if array.size else 1.0
    if max_value > 0:
        array /= max_value
    return array


def to_display_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray((clipped * 255).astype(np.uint8), mode="L")


def otsu_threshold(gray: np.ndarray) -> float:
    values = np.clip(gray, 0.0, 1.0)
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = values.size
    sum_total = np.dot(hist, np.arange(256))
    weight_bg = 0.0
    sum_bg = 0.0
    best_variance = -1.0
    best_threshold = 0
    for threshold in range(256):
        weight_bg += hist[threshold]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += threshold * hist[threshold]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return float(bin_edges[best_threshold])


def segment_breast(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove fundo e anotacoes mantendo o maior componente claro da mamografia."""
    threshold = max(otsu_threshold(gray), 0.03)
    mask = (gray > threshold).astype(np.uint8)

    if cv2 is not None:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            mask = (labels == largest).astype(np.uint8)
    else:
        # Fallback simples caso OpenCV nao esteja instalado.
        mask = largest_component_fallback(mask)

    segmented = gray * mask
    return segmented, mask.astype(np.float32)


def largest_component_fallback(mask: np.ndarray) -> np.ndarray:
    """Fallback sem OpenCV: preserva apenas o maior componente conectado."""
    visited = np.zeros(mask.shape, dtype=bool)
    best_component: list[tuple[int, int]] = []
    height, width = mask.shape
    for seed_y in range(height):
        for seed_x in range(width):
            if visited[seed_y, seed_x] or mask[seed_y, seed_x] == 0:
                continue
            component: list[tuple[int, int]] = []
            stack = [(seed_y, seed_x)]
            visited[seed_y, seed_x] = True
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if not visited[ny, nx] and mask[ny, nx] == 1:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) > len(best_component):
                best_component = component
    cleaned = np.zeros_like(mask)
    for y, x in best_component:
        cleaned[y, x] = 1
    return cleaned


class MammographyDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        task: str,
        train: bool,
        segmented: bool,
        image_size: int = 224,
    ) -> None:
        self.records = records
        self.task = task
        self.train = train
        self.segmented = segmented
        self.image_size = image_size
        self.samples: list[tuple[ImageRecord, int]] = []
        angles = ROTATION_ANGLES if train else [0]
        for record in records:
            for angle in angles:
                self.samples.append((record, angle))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record, angle = self.samples[index]
        gray = load_grayscale(record.path)
        if self.segmented:
            gray, _ = segment_breast(gray)
        image = to_display_image(gray).convert("RGB")
        if angle:
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        label = record.binary_index if self.task == "binary" else record.class_index
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def require_torch() -> None:
    if torch is None or models is None:
        raise RuntimeError(
            "PyTorch/torchvision nao estao instalados. Instale as dependencias de requirements.txt "
            "para treinar ResNet e EfficientNet."
        )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    require_torch()
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Modelo desconhecido: {model_name}")


def model_path(model_name: str, task: str, segmented: bool) -> Path:
    suffix = "segmentado" if segmented else "original"
    return MODEL_DIR / f"{model_name}_{task}_{suffix}.pt"


def train_model(
    records: list[ImageRecord],
    model_name: str,
    task: str,
    segmented: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    progress: Callable[[str], None],
) -> Path:
    require_torch()
    MODEL_DIR.mkdir(exist_ok=True)
    train_records = [record for record in records if record.split == "train"]
    num_classes = 2 if task == "binary" else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MammographyDataset(train_records, task=task, train=True, segmented=segmented)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
        progress(
            f"Epoca {epoch + 1}/{epochs}: loss={running_loss / max(total, 1):.4f} "
            f"acc={correct / max(total, 1):.4f}"
        )
    path = model_path(model_name, task, segmented)
    torch.save({"model_state": model.state_dict(), "task": task, "model_name": model_name}, path)
    progress(f"Treino concluido em {time.perf_counter() - start:.2f}s. Modelo salvo em {path}")
    return path


def load_trained_model(model_name: str, task: str, segmented: bool) -> nn.Module:
    require_torch()
    path = model_path(model_name, task, segmented)
    if not path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {path}. Treine antes de avaliar/classificar.")
    num_classes = 2 if task == "binary" else 4
    model = build_model(model_name, num_classes)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_model(
    records: list[ImageRecord],
    model_name: str,
    task: str,
    segmented: bool,
    batch_size: int,
) -> str:
    require_torch()
    test_records = [record for record in records if record.split == "test"]
    num_classes = 2 if task == "binary" else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_name, task, segmented).to(device)
    dataset = MammographyDataset(test_records, task=task, train=False, segmented=segmented)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            predictions = outputs.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.numpy().tolist())
    elapsed = time.perf_counter() - start
    confusion = confusion_matrix(y_true, y_pred, num_classes)
    if task == "binary":
        metrics = binary_metrics(confusion)
        return (
            f"Tempo de execucao: {elapsed:.2f}s\n"
            f"Matriz de confusao:\n{confusion}\n"
            f"Sensibilidade: {metrics['sensibilidade']:.4f}\n"
            f"Especificidade: {metrics['especificidade']:.4f}\n"
            f"Precisao: {metrics['precisao']:.4f}\n"
            f"Acuracia: {metrics['acuracia']:.4f}\n"
            f"F1: {metrics['f1']:.4f}"
        )
    sensitivity, specificity = multiclass_sensitivity_specificity(confusion)
    return (
        f"Tempo de execucao: {elapsed:.2f}s\n"
        f"Matriz de confusao (linhas=real, colunas=predito):\n{confusion}\n"
        f"Sensibilidade media: {sensitivity:.4f}\n"
        f"Especificidade media: {specificity:.4f}"
    )


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def binary_metrics(matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(matrix.sum(), 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)
    return {
        "sensibilidade": sensitivity,
        "especificidade": specificity,
        "precisao": precision,
        "acuracia": accuracy,
        "f1": f1,
    }


def multiclass_sensitivity_specificity(matrix: np.ndarray) -> tuple[float, float]:
    sensitivities = []
    specificities = []
    total = matrix.sum()
    for index in range(matrix.shape[0]):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        sensitivities.append(tp / max(tp + fn, 1))
        specificities.append(tn / max(tn + fp, 1))
    return float(np.mean(sensitivities)), float(np.mean(specificities))


def preprocess_for_model(path: Path, segmented: bool, image_size: int = 224) -> torch.Tensor:
    gray = load_grayscale(path)
    if segmented:
        gray, _ = segment_breast(gray)
    image = to_display_image(gray).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def grad_cam(model_name: str, task: str, segmented: bool, image_path: Path) -> tuple[str, Image.Image]:
    require_torch()
    model = load_trained_model(model_name, task, segmented)
    target_layer = model.layer4[-1] if model_name == "resnet18" else model.features[-1]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    model.eval()
    tensor = preprocess_for_model(image_path, segmented)
    output = model(tensor)
    predicted = int(output.argmax(dim=1).item())
    model.zero_grad()
    output[0, predicted].backward()
    handle_f.remove()
    handle_b.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).squeeze().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / max(float(cam.max()), 1e-8)
    base = to_display_image(load_grayscale(image_path)).convert("RGB").resize((224, 224))
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    if cv2 is not None:
        color = cv2.applyColorMap(np.asarray(heat), cv2.COLORMAP_JET)
        color = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    else:
        color = ImageOps.colorize(heat, black="black", white="red")
    overlay = Image.blend(base, color, alpha=0.45)
    label = "Baixa densidade (I+II)" if task == "binary" and predicted == 0 else "Alta densidade (III+IV)"
    if task == "four":
        label = f"{CLASS_NAMES[predicted]} - {BIRADS_LABELS[CLASS_NAMES[predicted]]}"
    return label, overlay


class MammographyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PAI - Segmentacao e classificacao mamografica")
        self.geometry("1180x760")
        self.records: list[ImageRecord] = []
        self.dataset_dir = DEFAULT_DATASET_DIR
        self.current_path: Path | None = None
        self.current_array: np.ndarray | None = None
        self.current_display: Image.Image | None = None
        self.tk_image = None
        self.message_queue: queue.Queue[str] = queue.Queue()

        self.model_var = tk.StringVar(value="resnet18")
        self.task_var = tk.StringVar(value="binary")
        self.segmented_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.epochs_var = tk.IntVar(value=3)
        self.batch_var = tk.IntVar(value=8)
        self.lr_var = tk.DoubleVar(value=0.001)

        self.build_layout()
        self.after(200, self.flush_messages)
        if self.dataset_dir.exists():
            self.load_dataset(self.dataset_dir)

    def build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        panel = ttk.Frame(self, padding=10)
        panel.grid(row=0, column=0, sticky="ns")

        ttk.Button(panel, text="Abrir imagem", command=self.open_image).grid(row=0, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="Selecionar dataset", command=self.choose_dataset).grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="Segmentar imagem", command=self.segment_current).grid(row=2, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=3, column=0, sticky="ew", pady=8)
        ttk.Label(panel, text="Modelo").grid(row=4, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.model_var, values=["resnet18", "efficientnet_b0"], state="readonly").grid(
            row=5, column=0, sticky="ew"
        )
        ttk.Label(panel, text="Tarefa").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(panel, textvariable=self.task_var, values=["binary", "four"], state="readonly").grid(
            row=7, column=0, sticky="ew"
        )
        ttk.Checkbutton(panel, text="Usar imagens segmentadas", variable=self.segmented_var).grid(
            row=8, column=0, sticky="w", pady=5
        )

        ttk.Label(panel, text="Epocas").grid(row=9, column=0, sticky="w")
        ttk.Spinbox(panel, from_=1, to=50, textvariable=self.epochs_var, width=8).grid(row=10, column=0, sticky="ew")
        ttk.Label(panel, text="Batch").grid(row=11, column=0, sticky="w")
        ttk.Spinbox(panel, from_=1, to=64, textvariable=self.batch_var, width=8).grid(row=12, column=0, sticky="ew")
        ttk.Label(panel, text="Learning rate").grid(row=13, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.lr_var).grid(row=14, column=0, sticky="ew")

        ttk.Button(panel, text="Treinar", command=self.train_selected).grid(row=15, column=0, sticky="ew", pady=(10, 3))
        ttk.Button(panel, text="Avaliar teste", command=self.evaluate_selected).grid(row=16, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="Grad-CAM", command=self.run_grad_cam).grid(row=17, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=18, column=0, sticky="ew", pady=8)
        ttk.Label(panel, text="Zoom").grid(row=19, column=0, sticky="w")
        ttk.Scale(panel, from_=0.2, to=3.0, variable=self.zoom_var, command=lambda _value: self.refresh_image()).grid(
            row=20, column=0, sticky="ew"
        )

        image_frame = ttk.Frame(self, padding=10)
        image_frame.grid(row=0, column=1, sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(image_frame, background="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.log = tk.Text(image_frame, height=10, wrap="word")
        self.log.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def log_message(self, message: str) -> None:
        self.message_queue.put(message)

    def flush_messages(self) -> None:
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.log.insert("end", message + "\n")
            self.log.see("end")
        self.after(200, self.flush_messages)

    def load_dataset(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.records = discover_dataset(dataset_dir)
        self.log_message(f"Dataset carregado: {dataset_dir}\n{summarize_records(self.records)}")

    def choose_dataset(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.dataset_dir if self.dataset_dir.exists() else Path.cwd()))
        if directory:
            self.load_dataset(Path(directory))

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("PNG", "*.png"), ("TIFF", "*.tif *.tiff")]
        )
        if not filename:
            return
        self.current_path = Path(filename)
        self.current_array = load_grayscale(self.current_path)
        self.current_display = to_display_image(self.current_array)
        self.zoom_var.set(1.0)
        self.refresh_image()
        self.log_message(f"Imagem aberta: {self.current_path}")

    def refresh_image(self) -> None:
        if self.current_display is None:
            return
        zoom = float(self.zoom_var.get())
        width = max(1, int(self.current_display.width * zoom))
        height = max(1, int(self.current_display.height * zoom))
        resized = self.current_display.resize((width, height), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def segment_current(self) -> None:
        if self.current_array is None:
            messagebox.showinfo("Segmentacao", "Abra uma imagem primeiro.")
            return
        segmented, _mask = segment_breast(self.current_array)
        self.current_array = segmented
        self.current_display = to_display_image(segmented)
        self.refresh_image()
        self.log_message("Segmentacao aplicada: fundo e anotacoes foram zerados quando possivel.")

    def ensure_records(self) -> bool:
        if not self.records:
            messagebox.showinfo("Dataset", "Selecione o diretorio Dataset/RCC antes.")
            return False
        return True

    def run_background(self, job: Callable[[], str | None]) -> None:
        def wrapper() -> None:
            try:
                result = job()
                if result:
                    self.log_message(result)
            except Exception as exc:
                self.log_message(f"ERRO: {exc}")

        threading.Thread(target=wrapper, daemon=True).start()

    def train_selected(self) -> None:
        if not self.ensure_records():
            return
        self.log_message("Iniciando treino. A interface continua responsiva; acompanhe o log.")

        def job() -> str:
            path = train_model(
                self.records,
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                int(self.epochs_var.get()),
                int(self.batch_var.get()),
                float(self.lr_var.get()),
                self.log_message,
            )
            return f"Modelo pronto: {path}"

        self.run_background(job)

    def evaluate_selected(self) -> None:
        if not self.ensure_records():
            return
        self.log_message("Avaliando conjunto de teste...")

        def job() -> str:
            return evaluate_model(
                self.records,
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                int(self.batch_var.get()),
            )

        self.run_background(job)

    def run_grad_cam(self) -> None:
        if self.current_path is None:
            messagebox.showinfo("Grad-CAM", "Abra uma imagem primeiro.")
            return
        self.log_message("Gerando Grad-CAM...")

        def job() -> str:
            label, overlay = grad_cam(
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                self.current_path,
            )
            self.current_display = overlay
            self.after(0, self.refresh_image)
            return f"Grad-CAM concluido. Classificacao: {label}"

        self.run_background(job)


def main() -> None:
    app = MammographyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
