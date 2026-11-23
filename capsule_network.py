import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

SEED = 42
NUM_CLASSES = 3
IMAGE_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TEST_IMAGES = 1500
CAPSULE_LAYOUT_VERSION = 2

DATASET_DIR = Path(
    "/content/drive/MyDrive/Software Engineering/6ºP/Redes Neurais"
)
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints"
RESULTS_DIR = DATASET_DIR / "resultados_teste"

CLASS_NAMES = ["Círculo", "Triângulo", "Quadrado"]
CLASS_TO_LABEL = {"circle": 0, "triangle": 1, "square": 2}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")
print("-" * 50)


# ============================================================
# 2. DATASET E SELEÇÃO BALANCEADA
# ============================================================

class GeometricShapesDataset(Dataset):
    """Carrega imagens PNG organizadas em pastas por classe."""

    CLASS_TO_LABEL = CLASS_TO_LABEL

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []

        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = self.root_dir / class_name
            image_paths = sorted(class_dir.glob("*.png"))

            if not image_paths:
                raise FileNotFoundError(
                    f"Nenhuma imagem PNG encontrada em: {class_dir}"
                )

            self.samples.extend(
                (image_path, label) for image_path in image_paths
            )

        if not self.samples:
            raise ValueError(f"Nenhuma imagem encontrada em {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as source:
            image = preprocess_image(source)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def create_balanced_subset(dataset, total_samples, seed=SEED):
    """Seleciona a mesma quantidade de imagens de cada classe."""
    num_classes = len(dataset.CLASS_TO_LABEL)

    if total_samples % num_classes != 0:
        raise ValueError(
            f"total_samples ({total_samples}) precisa ser divisível "
            f"por num_classes ({num_classes})."
        )

    samples_per_class = total_samples // num_classes
    rng = random.Random(seed)
    selected_indices = []

    for class_name, label in dataset.CLASS_TO_LABEL.items():
        class_indices = [
            index
            for index, (_, sample_label) in enumerate(dataset.samples)
            if sample_label == label
        ]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"A classe '{class_name}' tem {len(class_indices)} imagens; "
                f"são necessárias {samples_per_class}. "
                f"Verifique: {dataset.root_dir / class_name}"
            )

        selected_indices.extend(
            rng.sample(class_indices, samples_per_class)
        )

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


# ============================================================
# 3. ARQUITETURA CAPSULE NETWORK
# ============================================================

def squash(vectors, dim=-1):
    """Reduz o comprimento do vetor para [0, 1), preservando a direção."""
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = (
        squared_norm / (1.0 + squared_norm)
    ) / torch.sqrt(squared_norm + 1e-8)
    return scale * vectors


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels=256, capsule_channels=32, capsule_dim=8):
        super().__init__()
        self.capsule_channels = capsule_channels
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            capsule_channels * capsule_dim,
            kernel_size=9,
            stride=2
        )

    def forward(self, x):
        x = self.conv(x)
        batch_size, _, height, width = x.shape
        x = x.view(
            batch_size,
            self.capsule_channels,
            self.capsule_dim,
            height,
            width
        )
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return squash(x)


class DigitCapsules(nn.Module):
    def __init__(
        self,
        num_input_capsules=1152,
        input_dim=8,
        num_classes=3,
        output_dim=16,
        routing_iterations=3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(
            0.01 * torch.randn(
                1,
                num_input_capsules,
                num_classes,
                output_dim,
                input_dim
            )
        )

    def forward(self, x):
        batch_size = x.size(0)

        u_hat = torch.matmul(
            self.W,
            x[:, :, None, :, None]
        ).squeeze(-1)

        u_hat = u_hat.expand(batch_size, -1, -1, -1)

        b = torch.zeros(
            batch_size,
            u_hat.size(1),
            self.num_classes,
            device=x.device,
            dtype=x.dtype
        )

        for iteration in range(self.routing_iterations):
            coupling = F.softmax(b, dim=2)
            s = (coupling.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s)

            if iteration < self.routing_iterations - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v


class CapsuleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, stride=1),
            nn.ReLU()
        )
        self.primary_capsules = PrimaryCapsules()
        self.class_capsules = DigitCapsules()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.primary_capsules(x)
        x = self.class_capsules(x)
        return torch.norm(x, dim=-1)


def capsule_margin_loss(
    scores,
    labels,
    num_classes=NUM_CLASSES,
    m_plus=0.9,
    m_minus=0.1,
    lambda_=0.5
):
    targets = F.one_hot(labels, num_classes=num_classes).float()
    positive_loss = targets * F.relu(m_plus - scores).pow(2)
    negative_loss = (1 - targets) * F.relu(scores - m_minus).pow(2)
    return (positive_loss + lambda_ * negative_loss).sum(dim=1).mean()


# ============================================================
# 4. AVALIAÇÃO, CHECKPOINT E PREDIÇÃO
# ============================================================

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            scores = model(images)
            loss = capsule_margin_loss(scores, labels)

            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    if total == 0:
        raise ValueError("O carregador de avaliação está vazio.")

    return total_loss / total, 100.0 * correct / total


def save_checkpoint(model, optimizer, epoch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "class_to_label": CLASS_TO_LABEL,
            "image_size": IMAGE_SIZE,
            "capsule_layout_version": CAPSULE_LAYOUT_VERSION,
        },
        path
    )


def organize_test_predictions(model, dataset, loader, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / timestamp

    label_to_folder = {
        label: class_name
        for class_name, label in dataset.CLASS_TO_LABEL.items()
    }
    for class_name in label_to_folder.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    # loader.dataset pode ser um Subset; usamos seus índices para
    # localizar corretamente os arquivos originais.
    if isinstance(loader.dataset, Subset):
        sample_indices = loader.dataset.indices
    else:
        sample_indices = list(range(len(dataset)))

    counts = {class_name: 0 for class_name in label_to_folder.values()}
    sample_index = 0
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            scores = model(images.to(device))
            predictions = scores.argmax(dim=1).cpu().tolist()

            for predicted_label in predictions:
                source_path, _ = dataset.samples[sample_indices[sample_index]]
                class_name = label_to_folder[predicted_label]
                shutil.copy2(
                    source_path,
                    output_dir / class_name / source_path.name
                )
                counts[class_name] += 1
                sample_index += 1

    print(f"\nPrevisões copiadas para: {output_dir}")
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")

    return output_dir


def predict(model, image, true_label=None):
    """Recebe uma imagem no formato [1, 28, 28]."""
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        scores = model(image)
        probabilities = F.softmax(scores, dim=1)
        predicted_class = scores.argmax(dim=1).item()

    print("\nResultado da classificação")
    print("-" * 40)
    print(f"Classe prevista: {CLASS_NAMES[predicted_class]}")
    print(f"Pontuações das cápsulas: {scores[0].cpu().numpy().round(3)}")
    print(
        "Softmax (referência; não é necessariamente uma probabilidade "
        "calibrada):",
        probabilities[0].cpu().numpy().round(3)
    )

    if true_label is not None:
        print(f"Classe real: {CLASS_NAMES[int(true_label)]}")

    return predicted_class


def preprocess_image(source):
    """Converte uma imagem PIL para o formato usado pelo modelo."""
    image = source.convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR
    )
    image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)


def load_model(checkpoint_path=None):
    """Carrega o melhor modelo salvo para classificar imagens."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "capsule_network_best.pth"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            "Treine o modelo antes ou informe o caminho correto."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint inválido: {checkpoint_path}")
    if checkpoint.get("capsule_layout_version") != CAPSULE_LAYOUT_VERSION:
        raise CheckpointCompatibilityError(
            f"Checkpoint incompatível: {checkpoint_path}. "
            "O formato das cápsulas mudou; é necessário treinar novamente."
        )

    model = CapsuleNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class CheckpointCompatibilityError(ValueError):
    """Indica que checkpoint usa uma versão incompatível da arquitetura."""


def predict_image_file(model, image_path):
    """Classifica uma imagem de arquivo e mostra a forma prevista."""
    with Image.open(image_path) as source:
        image = preprocess_image(source)
    return predict(model, image)


def upload_and_predict(model=None):
    """Carrega ou treina o modelo e classifica imagens do Google Colab."""
    if model is None:
        checkpoint_path = CHECKPOINT_DIR / "capsule_network_best.pth"
        print(f"Procurando checkpoint em: {checkpoint_path}")
        if checkpoint_path.is_file():
            print("Checkpoint encontrado. Iniciando carregamento...")
            try:
                model = load_model(checkpoint_path)
            except CheckpointCompatibilityError as error:
                print(f"{error} Treinando novamente...")
                model, _, _ = train()
            else:
                print("Checkpoint carregado com sucesso.")
        else:
            available_checkpoints = sorted(
                path.name for path in CHECKPOINT_DIR.glob("*.pth")
            )
            print(f"Checkpoint esperado não encontrado: {checkpoint_path}")
            print(
                "Outros arquivos .pth encontrados: "
                f"{available_checkpoints or 'nenhum'}"
            )
            print("Iniciando treinamento porque o checkpoint esperado falta.")
            model, _, _ = train()
            print("Treinamento concluído.")
    else:
        print(
            "Modelo recebido como argumento. Não vou procurar checkpoint "
            "nem iniciar treinamento."
        )

    from google.colab import files

    results = {}
    while True:
        print("\nSelecione uma ou mais imagens para classificar.")
        uploaded_files = files.upload()
        if not uploaded_files:
            print("Nenhuma imagem enviada. Encerrando os testes.")
            break

        for filename in uploaded_files:
            predicted_class = predict_image_file(model, filename)
            results[filename] = CLASS_NAMES[predicted_class]

        repeat = input("\nDeseja enviar mais imagens? (s/n): ").strip().lower()
        if repeat not in {"s", "sim"}:
            break

    return results


# ============================================================
# 5. TESTE DE DIAGNÓSTICO
# ============================================================

def diagnostic_test(model, train_loader):
    print("\nTeste de diagnóstico: carregamento, forward e backward")
    print("-" * 50)

    start = __import__("time").time()
    images, labels = next(iter(train_loader))
    print(
        f"Lote carregado em {__import__('time').time() - start:.2f}s",
        images.shape,
        labels.shape
    )

    model.train()
    model.zero_grad(set_to_none=True)
    images = images.to(device)
    labels = labels.to(device)

    start = __import__("time").time()
    scores = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"Forward concluído em {__import__('time').time() - start:.2f}s",
        scores.shape
    )

    loss = capsule_margin_loss(scores, labels)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss não finita: {loss.item()}")

    start = __import__("time").time()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    has_gradients = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"Backward concluído em {__import__('time').time() - start:.2f}s"
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradientes calculados: {has_gradients}")

    model.zero_grad(set_to_none=True)
    if not has_gradients:
        raise RuntimeError("Nenhum gradiente foi calculado.")

    print("Diagnóstico concluído.")


# ============================================================
# 6. TREINAMENTO COMPLETO
# ============================================================

def train():
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            "Não encontrei as pastas train/test no Google Drive. "
            "Monte o Drive e confira DATASET_DIR."
        )

    train_dataset = GeometricShapesDataset(TRAIN_DIR)
    full_test_dataset = GeometricShapesDataset(TEST_DIR)

    # O teste usa 300 imagens balanceadas: 100 por classe.
    test_subset = create_balanced_subset(
        full_test_dataset,
        TEST_IMAGES,
        seed=SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    print(f"Imagens de treinamento: {len(train_dataset)}")
    print(f"Imagens de teste selecionadas: {len(test_subset)}")
    print(f"Lotes de treino: {len(train_loader)}")
    print(f"Lotes de teste: {len(test_loader)}")
    print("-" * 50)

    model = CapsuleNetwork().to(device)

    # Executa o teste antes de iniciar as épocas.
    diagnostic_test(model, train_loader)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "capsule_network_latest.pth"
    best_accuracy = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = capsule_margin_loss(scores, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss não finita na época {epoch + 1}, "
                    f"lote {batch_index}: {loss.item()}"
                )

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        test_loss, test_accuracy = evaluate(model, test_loader)

        print(
            f"Época {epoch + 1:02d}/{EPOCHS} | "
            f"Loss treino: {train_loss:.4f} | "
            f"Loss teste: {test_loss:.4f} | "
            f"Acurácia teste: {test_accuracy:.2f}%"
        )

        save_checkpoint(
            model, optimizer, epoch + 1, checkpoint_path
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_path = CHECKPOINT_DIR / "capsule_network_best.pth"
            save_checkpoint(
                model, optimizer, epoch + 1, best_path
            )

    print("-" * 50)
    print(f"Último checkpoint salvo em: {checkpoint_path}")
    print(f"Melhor acurácia de teste: {best_accuracy:.2f}%")

    organize_test_predictions(
        model,
        full_test_dataset,
        test_loader,
        RESULTS_DIR
    )

    # Retorna também o dataset completo de teste, útil para inspeção.
    return model, test_loader, full_test_dataset


# ============================================================
# 7. EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    upload_and_predict()