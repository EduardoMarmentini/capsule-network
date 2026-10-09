
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


SEED = 42
NUM_CLASSES = 3
IMAGE_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TEST_IMAGES = 300
DATASET_DIR = Path(
    "/content/drive/MyDrive/Software Engineering/6ºP/Redes Neurais"
)
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

CLASS_NAMES = ["Círculo", "Triângulo", "Quadrado"]

torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo utilizado: {device}")
print("-" * 40)


class GeometricShapesDataset(Dataset):
    """Carrega imagens PNG organizadas em pastas por classe."""

    CLASS_TO_LABEL = {
        "circle": 0,
        "triangle": 1,
        "square": 2,
    }

    def __init__(self, root_dir):
        self.samples = []

        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = Path(root_dir) / class_name
            image_paths = sorted(class_dir.glob("*.png"))
            if not image_paths:
                raise FileNotFoundError(
                    f"Nenhuma imagem PNG encontrada em: {class_dir}"
                )

            self.samples.extend(
                (image_path, label) for image_path in image_paths
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BILINEAR
            )

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def create_balanced_subset(dataset, total_samples, seed=SEED):
    num_classes = len(dataset.CLASS_TO_LABEL)
    if total_samples % num_classes != 0:
        raise ValueError(
            "O total de amostras deve ser divisível pelo número de classes."
        )

    samples_per_class = total_samples // num_classes
    rng = random.Random(seed)
    selected_indices = []

    for label in dataset.CLASS_TO_LABEL.values():
        class_indices = [
            index
            for index, (_, sample_label) in enumerate(dataset.samples)
            if sample_label == label
        ]
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"A classe {label} não possui amostras suficientes."
            )

        selected_indices.extend(
            rng.sample(class_indices, samples_per_class)
        )

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def squash(vectors, dim=-1):
    """
    Reduz o comprimento do vetor para o intervalo [0, 1).

    A direção do vetor é preservada.
    """
    squared_norm = (vectors ** 2).sum(
        dim=dim, keepdim=True
    )

    scale = (
        squared_norm / (1.0 + squared_norm)
    ) / torch.sqrt(squared_norm + 1e-8)

    return scale * vectors


class PrimaryCapsules(nn.Module):
    def __init__(
        self,
        in_channels=256,
        capsule_channels=32,
        capsule_dim=8
    ):
        super().__init__()

        self.capsule_dim = capsule_dim

        self.conv = nn.Conv2d(
            in_channels,
            capsule_channels * capsule_dim,
            kernel_size=9,
            stride=2
        )

    def forward(self, x):
        x = self.conv(x)

        # [B, 256, 6, 6] -> [B, 1152, 8]
        batch_size = x.size(0)

        x = x.view(
            batch_size,
            self.capsule_dim,
            -1
        ).transpose(1, 2)

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

        # Matriz de transformação de cada cápsula
        # de entrada para cada classe.
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

        # Previsões das cápsulas de entrada.
        u_hat = torch.matmul(
            self.W,
            x[:, :, None, :, None]
        ).squeeze(-1)

        # [B, cápsulas de entrada, classes, dimensão]
        u_hat = u_hat.expand(
            batch_size, -1, -1, -1
        )

        # Logits de roteamento.
        b = torch.zeros(
            batch_size,
            u_hat.size(1),
            self.num_classes,
            device=x.device,
            dtype=x.dtype
        )

        for iteration in range(
            self.routing_iterations
        ):
            # Pesos de conexão entre cápsulas.
            coupling = F.softmax(b, dim=2)

            # Combina as previsões para cada classe.
            s = (
                coupling.unsqueeze(-1) * u_hat
            ).sum(dim=1)

            # Vetores das cápsulas de classe.
            v = squash(s)

            if iteration < self.routing_iterations - 1:
                # Reforça as previsões concordantes.
                agreement = (
                    u_hat * v.unsqueeze(1)
                ).sum(dim=-1)

                b = b + agreement

        return v


class CapsuleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, 256,
                kernel_size=9,
                stride=1
            ),
            nn.ReLU()
        )

        self.primary_capsules = PrimaryCapsules()

        self.class_capsules = DigitCapsules()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.primary_capsules(x)
        x = self.class_capsules(x)

        # O comprimento de cada vetor representa
        # a evidência para uma classe.
        return torch.norm(x, dim=-1)


def capsule_margin_loss(
    scores,
    labels,
    num_classes=3,
    m_plus=0.9,
    m_minus=0.1,
    lambda_=0.5
):
    """
    Incentiva a cápsula da classe correta a ter
    comprimento elevado e as demais, comprimento baixo.
    """
    targets = F.one_hot(
        labels, num_classes=num_classes
    ).float()

    positive_loss = targets * F.relu(
        m_plus - scores
    ).pow(2)

    negative_loss = (
        (1 - targets)
        * F.relu(scores - m_minus).pow(2)
    )

    loss = positive_loss + lambda_ * negative_loss

    return loss.sum(dim=1).mean()


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

            loss = capsule_margin_loss(
                scores, labels
            )

            predictions = scores.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


def organize_test_predictions(model, dataset, loader, output_dir):
    output_dir = Path(output_dir) / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    label_to_folder = {
        label: class_name
        for class_name, label in dataset.CLASS_TO_LABEL.items()
    }

    for class_name in label_to_folder.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    counts = {class_name: 0 for class_name in label_to_folder.values()}
    sample_indices = getattr(
        loader.dataset,
        "indices",
        range(len(dataset))
    )
    sample_index = 0
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            scores = model(images.to(device))
            predictions = scores.argmax(dim=1).cpu().tolist()

            for predicted_label in predictions:
                source_path, _ = dataset.samples[
                    sample_indices[sample_index]
                ]
                class_name = label_to_folder[predicted_label]
                shutil.copy2(
                    source_path,
                    output_dir / class_name / source_path.name
                )
                counts[class_name] += 1
                sample_index += 1

    print(f"\nImagens classificadas copiadas para: {output_dir}")
    print("-" * 40)
    for class_name, count in counts.items():
        print(f"{CLASS_NAMES[dataset.CLASS_TO_LABEL[class_name]]}: {count}")


def train():
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            "Não encontrei as pastas train/test no Google Drive. "
            "Monte o Drive no Colab e confira DATASET_DIR."
        )

    train_dataset = GeometricShapesDataset(TRAIN_DIR)
    test_dataset = GeometricShapesDataset(TEST_DIR)
    test_subset = create_balanced_subset(
        test_dataset,
        TEST_IMAGES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CapsuleNetwork().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"Imagens de treinamento: {len(train_dataset)}")
    print("-" * 40)
    print(f"Imagens de teste: {len(test_subset)}")
    print("-" * 40)

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            scores = model(images)

            loss = capsule_margin_loss(
                scores, labels
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * labels.size(0)
            )

        train_loss = (
            total_loss / len(train_dataset)
        )

        test_loss, test_accuracy = evaluate(
            model, test_loader
        )

        print(
            f"Época {epoch + 1:02d}/{EPOCHS} | "
            f"Loss treino: {train_loss:.4f} | "
            f"Loss teste: {test_loss:.4f} | "
            f"Acurácia teste: {test_accuracy:.2f}%"
        )
        print("-" * 40)
    organize_test_predictions(
        model,
        test_dataset,
        test_loader,
        DATASET_DIR / "resultados_teste"
    )

    return model, test_loader


def predict(model, image, true_label=None):
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)

        scores = model(image)
        probabilities = F.softmax(scores, dim=1)

        predicted_class = scores.argmax(dim=1).item()

    print("\nResultado da classificação")
    print("-" * 40)
    print(f"Classe prevista: {CLASS_NAMES[predicted_class]}")
    print("-" * 40)
    print(
        f"Pontuações das cápsulas: "
        f"{scores[0].cpu().numpy().round(3)}"
    )
    print("-" * 40)
    print(
        f"Softmax (referência): "
        f"{probabilities[0].cpu().numpy().round(3)}"
    )
    print("-" * 40)

    if true_label is not None:
        print(f"Classe real: {CLASS_NAMES[true_label]}")
        print("-" * 40)

    return predicted_class


if __name__ == "__main__":
    model, test_loader = train()

    rng = random.Random(SEED + 1)
    sample_index = rng.randrange(len(test_loader.dataset))
    image, label = test_loader.dataset[sample_index]

    predict(
        model,
        image[0],
        true_label=label[0].item()
    )