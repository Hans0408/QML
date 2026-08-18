"""
Dynamic circuit-parameterized MPS -> VQC Bars-and-Stripes pipeline.

The script first trains a circuit-parameterized MPS, then transfers its
trained parameters directly into a dynamically constructed VQC.

Changing IMAGE_SIZE or BOND_DIM updates both models automatically.

Requirements:
    pip install torch pennylane

Important:
    BOND_DIM must be a power of 2.

Classification rule:
    P(readout wire = 0) -> class 0
    P(readout wire = 1) -> class 1
"""

import copy
import math
import random
from dataclasses import dataclass

import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Configuration
# ============================================================

SEEDS = [45]

IMAGE_SIZE = 2
BOND_DIM = 4

PHYSICAL_DIM = 2
OUTPUT_DIM = 2

MPS_CIRCUIT_LAYERS = 6

MPS_EPOCHS = 100
MPS_LEARNING_RATE = 1e-3
MPS_BATCH_SIZE = 32

VQC_EPOCHS = 30
VQC_LEARNING_RATE = 1e-3
VQC_BATCH_SIZE = 16

VQC_RANDOM_SCALE = 2 * torch.pi

N_TRAIN_SAMPLES = 800
N_TEST_SAMPLES = 200
BAS_NOISE_PROBABILITY = 0.05

TRAINING_SHOTS = None
EVALUATION_SHOTS = 1000
EPSILON = 1e-7

READOUT_WIRE_MODE = "mps_output"

USE_NEAR_IDENTITY_EXTENSION = True
EXTRA_GATE_LAYERS = 1
EXTRA_GATE_SCALE = 1e-3
EXTRA_GATE_TOPOLOGY = "all_to_all"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def bond_dim_to_qubits(bond_dim: int) -> int:
    if not is_power_of_two(bond_dim):
        raise ValueError(
            f"BOND_DIM must be a power of 2. Got BOND_DIM={bond_dim}."
        )
    return int(math.log2(bond_dim))


def number_of_so_angles(matrix_dim: int) -> int:
    return matrix_dim * (matrix_dim - 1) // 2


# ============================================================
# Data
# ============================================================

class FeatureDataset(Dataset):
    """Convert image pixels into cosine/sine MPS features."""

    def __init__(self, image_dataset):
        self.image_dataset = image_dataset

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        image, label = self.image_dataset[index]
        pixels = image.reshape(-1).to(torch.float64)
        angles = 0.5 * torch.pi * pixels

        features = torch.stack(
            [torch.cos(angles), torch.sin(angles)],
            dim=-1,
        )
        return features, int(label)


class BarsAndStripesDataset(Dataset):
    """
    Balanced synthetic Bars-and-Stripes dataset.

    Labels:
        0 -> vertical bars
        1 -> horizontal stripes

    Uniform patterns are excluded because they belong to both classes.
    """

    def __init__(
        self,
        image_size: int,
        num_samples: int,
        seed: int,
        noise_probability: float = 0.0,
    ):
        if image_size < 2:
            raise ValueError("Bars-and-Stripes requires image_size >= 2.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be in [0, 1].")

        self.num_samples = num_samples

        generator = torch.Generator().manual_seed(seed)
        images = []
        labels = []

        class_counts = [
            num_samples // 2,
            num_samples - num_samples // 2,
        ]

        for label, count in enumerate(class_counts):
            for _ in range(count):
                while True:
                    pattern = torch.randint(
                        0,
                        2,
                        (image_size,),
                        generator=generator,
                        dtype=torch.int64,
                    )
                    if not torch.all(pattern == pattern[0]):
                        break

                if label == 0:
                    image = pattern.unsqueeze(0).repeat(image_size, 1)
                else:
                    image = pattern.unsqueeze(1).repeat(1, image_size)

                image = image.to(torch.float64)

                if noise_probability > 0.0:
                    flips = (
                        torch.rand(
                            image.shape,
                            generator=generator,
                            dtype=torch.float64,
                        )
                        < noise_probability
                    )
                    image = torch.where(flips, 1.0 - image, image)

                images.append(image.unsqueeze(0))
                labels.append(label)

        permutation = torch.randperm(num_samples, generator=generator)
        self.images = torch.stack(images)[permutation]
        self.targets = torch.tensor(labels, dtype=torch.long)[permutation]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


def make_bars_and_stripes_image_datasets(
    image_size: int,
    train_size: int,
    test_size: int,
    seed: int,
    noise_probability: float,
):
    train_dataset = BarsAndStripesDataset(
        image_size=image_size,
        num_samples=train_size,
        seed=seed,
        noise_probability=noise_probability,
    )
    test_dataset = BarsAndStripesDataset(
        image_size=image_size,
        num_samples=test_size,
        seed=seed + 1,
        noise_probability=noise_probability,
    )
    return train_dataset, test_dataset


def make_loaders(image_size: int, seed: int):
    train_images, test_images = make_bars_and_stripes_image_datasets(
        image_size=image_size,
        train_size=N_TRAIN_SAMPLES,
        test_size=N_TEST_SAMPLES,
        seed=seed,
        noise_probability=BAS_NOISE_PROBABILITY,
    )

    train_features = FeatureDataset(train_images)
    test_features = FeatureDataset(test_images)

    mps_train_loader = DataLoader(
        train_features,
        batch_size=MPS_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    mps_test_loader = DataLoader(
        test_features,
        batch_size=MPS_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    vqc_train_loader = DataLoader(
        train_images,
        batch_size=VQC_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    vqc_test_loader = DataLoader(
        test_images,
        batch_size=VQC_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        mps_train_loader,
        mps_test_loader,
        vqc_train_loader,
        vqc_test_loader,
    )


# ============================================================
# SO rotations for optional extra gates
# ============================================================

def lexicographic_rotation_pairs(matrix_dim: int):
    return [
        (i, j)
        for i in range(matrix_dim)
        for j in range(i + 1, matrix_dim)
    ]


def xor_grouped_rotation_pairs(matrix_dim: int):
    if not is_power_of_two(matrix_dim):
        raise ValueError(
            "XOR grouping requires matrix_dim to be a power of two. "
            f"Got matrix_dim={matrix_dim}."
        )

    gray_masks = [k ^ (k >> 1) for k in range(1, matrix_dim)]
    grouped_pairs = []

    for mask in gray_masks:
        for i in range(matrix_dim):
            j = i ^ mask
            if i < j:
                grouped_pairs.append((i, j))

    expected = number_of_so_angles(matrix_dim)
    if len(grouped_pairs) != expected or len(set(grouped_pairs)) != expected:
        raise RuntimeError("Invalid XOR-grouped rotation-pair construction.")

    return grouped_pairs


def _left_apply_plane_rotation(
    matrix: torch.Tensor,
    i: int,
    j: int,
    theta: torch.Tensor,
):
    c = torch.cos(theta)
    s = torch.sin(theta)

    row_i = matrix[..., i, :].clone()
    row_j = matrix[..., j, :].clone()

    matrix[..., i, :] = c[..., None] * row_i - s[..., None] * row_j
    matrix[..., j, :] = s[..., None] * row_i + c[..., None] * row_j


def rotation_matrix_nd(
    n: int,
    thetas: torch.Tensor,
) -> torch.Tensor:
    expected = number_of_so_angles(n)

    if thetas.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} angles for SO({n}), "
            f"got {thetas.shape[-1]}."
        )

    matrix = (
        torch.eye(
            n,
            dtype=thetas.dtype,
            device=thetas.device,
        )
        .expand(*thetas.shape[:-1], n, n)
        .clone()
    )

    original_pairs = lexicographic_rotation_pairs(n)
    theta_index = {pair: k for k, pair in enumerate(original_pairs)}
    factor_order = xor_grouped_rotation_pairs(n)

    for i, j in reversed(factor_order):
        k = theta_index[(i, j)]
        _left_apply_plane_rotation(matrix, i, j, thetas[..., k])

    return matrix


# ============================================================
# Efficient circuit-family matrices
# ============================================================

def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2
    c = torch.cos(half)
    s = torch.sin(half)

    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )


def _kron_all(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def _single_qubit_operator(
    gate,
    wire: int,
    num_qubits: int,
):
    identity = torch.eye(
        2,
        dtype=gate.dtype,
        device=gate.device,
    )
    factors = [identity for _ in range(num_qubits)]
    factors[wire] = gate
    return _kron_all(factors)


def _controlled_ry_operator(
    theta: torch.Tensor,
    control: int,
    target: int,
    num_qubits: int,
):
    dtype = theta.dtype
    device = theta.device

    identity = torch.eye(2, dtype=dtype, device=device)
    p0 = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    p1 = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    rotation = ry_matrix(theta)

    factors_zero = [identity for _ in range(num_qubits)]
    factors_one = [identity for _ in range(num_qubits)]

    factors_zero[control] = p0
    factors_one[control] = p1
    factors_one[target] = rotation

    return _kron_all(factors_zero) + _kron_all(factors_one)


def _efficient_circuit_matrix_single(
    num_qubits: int,
    thetas: torch.Tensor,
) -> torch.Tensor:
    if thetas.ndim != 2:
        raise ValueError(
            "Expected theta shape [layers, params_per_layer], "
            f"got {tuple(thetas.shape)}."
        )

    num_layers, params_per_layer = thetas.shape
    expected = 2 * num_qubits - 1

    if params_per_layer != expected:
        raise ValueError(
            f"Expected {expected} parameters per layer "
            f"for {num_qubits} qubits, got {params_per_layer}."
        )

    dimension = 2 ** num_qubits
    unitary = torch.eye(
        dimension,
        dtype=thetas.dtype,
        device=thetas.device,
    )

    for layer in range(num_layers):
        local = torch.eye(
            dimension,
            dtype=thetas.dtype,
            device=thetas.device,
        )

        for wire in range(num_qubits):
            local = (
                _single_qubit_operator(
                    ry_matrix(thetas[layer, wire]),
                    wire=wire,
                    num_qubits=num_qubits,
                )
                @ local
            )

        entangler = torch.eye(
            dimension,
            dtype=thetas.dtype,
            device=thetas.device,
        )
        edge_angles = thetas[layer, num_qubits:]

        if layer % 2 == 0:
            edges = [
                (wire, wire + 1)
                for wire in range(num_qubits - 1)
            ]
        else:
            edges = [
                (wire + 1, wire)
                for wire in range(num_qubits - 2, -1, -1)
            ]

        for edge_index, (control, target) in enumerate(edges):
            entangler = (
                _controlled_ry_operator(
                    edge_angles[edge_index],
                    control=control,
                    target=target,
                    num_qubits=num_qubits,
                )
                @ entangler
            )

        unitary = entangler @ local @ unitary

    return unitary


def efficient_circuit_matrix(
    num_qubits: int,
    thetas: torch.Tensor,
) -> torch.Tensor:
    if thetas.ndim == 2:
        return _efficient_circuit_matrix_single(
            num_qubits,
            thetas,
        )

    batch_shape = thetas.shape[:-2]
    flat = thetas.reshape(-1, *thetas.shape[-2:])

    matrices = [
        _efficient_circuit_matrix_single(
            num_qubits,
            parameter_block,
        )
        for parameter_block in flat
    ]

    result = torch.stack(matrices, dim=0)

    return result.reshape(
        *batch_shape,
        2 ** num_qubits,
        2 ** num_qubits,
    )


# ============================================================
# MPS
# ============================================================

class MPS1(nn.Module):
    """Circuit-parameterized real MPS classifier."""

    def __init__(
        self,
        num_sites: int,
        bond_dim: int,
        physical_dim: int = 2,
        output_dim: int = 2,
        circuit_layers: int = MPS_CIRCUIT_LAYERS,
    ):
        super().__init__()

        if num_sites < 2:
            raise ValueError("num_sites must be at least 2.")
        if physical_dim != 2:
            raise ValueError(
                "The efficient circuit family assumes physical_dim=2."
            )

        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.physical_dim = physical_dim
        self.output_dim = output_dim
        self.circuit_layers = circuit_layers

        self.bond_qubits = bond_dim_to_qubits(bond_dim)
        self.bulk_qubits = self.bond_qubits + 1

        first_params_per_layer = 2 * self.bond_qubits - 1
        bulk_params_per_layer = 2 * self.bulk_qubits - 1

        self.first_thetas = nn.Parameter(
            torch.empty(
                circuit_layers,
                first_params_per_layer,
                dtype=torch.float64,
            )
        )
        self.middle_thetas = nn.Parameter(
            torch.empty(
                num_sites - 2,
                circuit_layers,
                bulk_params_per_layer,
                dtype=torch.float64,
            )
        )
        self.last_thetas = nn.Parameter(
            torch.empty(
                circuit_layers,
                bulk_params_per_layer,
                dtype=torch.float64,
            )
        )

        self.reset_parameters(scale=0.01)

    def reset_parameters(self, scale: float = 0.01):
        with torch.no_grad():
            self.first_thetas.uniform_(-scale, scale)
            self.middle_thetas.uniform_(-scale, scale)
            self.last_thetas.uniform_(-scale, scale)

    def first_tensor(self):
        matrix = efficient_circuit_matrix(
            self.bond_qubits,
            self.first_thetas,
        )
        return matrix[:, :self.physical_dim]

    def middle_tensors(self):
        matrices = efficient_circuit_matrix(
            self.bulk_qubits,
            self.middle_thetas,
        )
        matrices = matrices[:, :self.bond_dim, :]

        return matrices.reshape(
            self.num_sites - 2,
            self.bond_dim,
            self.bond_dim,
            self.physical_dim,
        )

    def last_tensor(self):
        matrix = efficient_circuit_matrix(
            self.bulk_qubits,
            self.last_thetas,
        )

        return matrix.reshape(
            self.output_dim,
            self.bond_dim,
            self.bond_dim,
            self.physical_dim,
        )

    def amplitude_outputs(self, features):
        first = self.first_tensor()
        middles = self.middle_tensors()
        last = self.last_tensor()

        state = torch.einsum(
            "np,bp->bn",
            first,
            features[:, 0, :],
        )

        for site in range(1, self.num_sites - 1):
            state = torch.einsum(
                "nlp,bl,bp->bn",
                middles[site - 1],
                state,
                features[:, site, :],
            )

        return torch.einsum(
            "oelp,bl,bp->boe",
            last,
            state,
            features[:, -1, :],
        )

    def forward(self, features):
        amplitudes = self.amplitude_outputs(features)
        class_weights = amplitudes.abs().square().sum(dim=-1)

        return class_weights / class_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(EPSILON)


# ============================================================
# MPS training
# ============================================================

def compute_gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0,
) -> float:
    gradient_norms = [
        parameter.grad.detach().norm(norm_type)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    if not gradient_norms:
        return 0.0

    return torch.stack(gradient_norms).norm(norm_type).item()


def train_one_mps_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_gradient_norm = 0.0
    maximum_gradient_norm = 0.0
    number_of_batches = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        probabilities = model(features)
        log_probabilities = torch.log(
            probabilities.clamp_min(EPSILON)
        )
        loss = criterion(log_probabilities, labels)

        loss.backward()

        gradient_norm = compute_gradient_norm(model)
        total_gradient_norm += gradient_norm
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            gradient_norm,
        )
        number_of_batches += 1

        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            probabilities.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_gradient_norm / max(number_of_batches, 1),
        maximum_gradient_norm,
    )


@torch.no_grad()
def evaluate_mps(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        probabilities = model(features)
        log_probabilities = torch.log(
            probabilities.clamp_min(EPSILON)
        )
        loss = criterion(log_probabilities, labels)

        predictions = probabilities.argmax(dim=1)
        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def train_mps_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    device,
):
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "average_gradient_norm": [],
        "maximum_gradient_norm": [],
    }

    best_train_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        (
            train_loss,
            train_accuracy,
            grad_avg,
            grad_max,
        ) = train_one_mps_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        test_loss, test_accuracy = evaluate_mps(
            model,
            test_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["average_gradient_norm"].append(grad_avg)
        history["maximum_gradient_norm"].append(grad_max)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[MPS] Epoch {epoch:03d}/{num_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"train acc: {train_accuracy:.2%} | "
            f"test loss: {test_loss:.6f} | "
            f"test acc: {test_accuracy:.2%} | "
            f"grad avg: {grad_avg:.3e} | "
            f"grad max: {grad_max:.3e}"
        )

    model.load_state_dict(best_state)

    print(
        f"\n[MPS] Smallest train loss: "
        f"{best_train_loss:.6f} at epoch {best_epoch}\n"
    )

    return model, history


# ============================================================
# VQC geometry
# ============================================================

@dataclass
class DynamicVQCGeometry:
    image_size: int
    num_sites: int
    bond_dim: int
    bond_qubits: int
    n_qubits: int
    fixed_zero_wires: list
    image_wires: list
    first_wires: list
    middle_wires: list
    last_wires: list


def build_vqc_geometry(
    image_size: int,
    bond_dim: int,
) -> DynamicVQCGeometry:
    num_sites = image_size * image_size
    bond_qubits = bond_dim_to_qubits(bond_dim)
    n_qubits = num_sites + bond_qubits - 1

    fixed_zero_wires = list(range(bond_qubits - 1))
    image_wires = list(range(bond_qubits - 1, n_qubits))
    first_wires = list(range(bond_qubits))

    middle_wires = [
        list(range(site - 1, site + bond_qubits))
        for site in range(1, num_sites - 1)
    ]

    last_wires = list(
        range(
            num_sites - 2,
            num_sites + bond_qubits - 1,
        )
    )

    return DynamicVQCGeometry(
        image_size=image_size,
        num_sites=num_sites,
        bond_dim=bond_dim,
        bond_qubits=bond_qubits,
        n_qubits=n_qubits,
        fixed_zero_wires=fixed_zero_wires,
        image_wires=image_wires,
        first_wires=first_wires,
        middle_wires=middle_wires,
        last_wires=last_wires,
    )


# ============================================================
# VQC
# ============================================================

def create_random_vqc_thetas(
    image_size: int,
    bond_dim: int,
    scale: float,
    circuit_layers: int = MPS_CIRCUIT_LAYERS,
):
    num_sites = image_size * image_size
    bond_qubits = bond_dim_to_qubits(bond_dim)
    bulk_qubits = bond_qubits + 1

    first_params_per_layer = 2 * bond_qubits - 1
    bulk_params_per_layer = 2 * bulk_qubits - 1

    first_thetas = scale * torch.randn(
        circuit_layers,
        first_params_per_layer,
        dtype=torch.float64,
    )
    middle_thetas = scale * torch.randn(
        num_sites - 2,
        circuit_layers,
        bulk_params_per_layer,
        dtype=torch.float64,
    )
    last_thetas = scale * torch.randn(
        circuit_layers,
        bulk_params_per_layer,
        dtype=torch.float64,
    )

    return first_thetas, middle_thetas, last_thetas


def build_extra_gate_pairs(
    n_qubits: int,
    topology: str,
):
    if topology == "all_to_all":
        return [
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        ]

    if topology == "linear":
        return [
            (i, i + 1)
            for i in range(n_qubits - 1)
        ]

    if topology == "nonlocal":
        return [
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 2, n_qubits)
        ]

    raise ValueError(
        "EXTRA_GATE_TOPOLOGY must be "
        "'all_to_all', 'linear', or 'nonlocal'."
    )


def create_near_identity_extra_thetas(
    n_qubits: int,
    layers: int,
    scale: float,
    topology: str,
):
    pairs = build_extra_gate_pairs(
        n_qubits,
        topology,
    )

    thetas = scale * torch.randn(
        layers,
        len(pairs),
        number_of_so_angles(4),
        dtype=torch.float64,
    )

    return pairs, thetas


class MPSInitializedDynamicVQC(nn.Module):
    """Dynamic VQC initialized from circuit-MPS parameters."""

    def __init__(
        self,
        image_size: int,
        bond_dim: int,
        first_thetas: torch.Tensor,
        middle_thetas: torch.Tensor,
        last_thetas: torch.Tensor,
    ):
        super().__init__()

        self.geometry = build_vqc_geometry(
            image_size,
            bond_dim,
        )

        self.image_size = image_size
        self.num_sites = self.geometry.num_sites
        self.bond_dim = bond_dim
        self.bond_qubits = self.geometry.bond_qubits
        self.n_qubits = self.geometry.n_qubits

        self.circuit_layers = first_thetas.shape[0]

        first_params_per_layer = 2 * self.bond_qubits - 1
        bulk_params_per_layer = 2 * (self.bond_qubits + 1) - 1

        expected_first_shape = (
            self.circuit_layers,
            first_params_per_layer,
        )
        expected_middle_shape = (
            self.num_sites - 2,
            self.circuit_layers,
            bulk_params_per_layer,
        )
        expected_last_shape = (
            self.circuit_layers,
            bulk_params_per_layer,
        )

        if tuple(first_thetas.shape) != expected_first_shape:
            raise ValueError(
                f"Expected first_thetas shape "
                f"{expected_first_shape}, "
                f"got {tuple(first_thetas.shape)}."
            )

        if tuple(middle_thetas.shape) != expected_middle_shape:
            raise ValueError(
                f"Expected middle_thetas shape "
                f"{expected_middle_shape}, "
                f"got {tuple(middle_thetas.shape)}."
            )

        if tuple(last_thetas.shape) != expected_last_shape:
            raise ValueError(
                f"Expected last_thetas shape "
                f"{expected_last_shape}, "
                f"got {tuple(last_thetas.shape)}."
            )

        self.first_thetas = nn.Parameter(
            first_thetas.detach().clone().to(torch.float64)
        )
        self.middle_thetas = nn.Parameter(
            middle_thetas.detach().clone().to(torch.float64)
        )
        self.last_thetas = nn.Parameter(
            last_thetas.detach().clone().to(torch.float64)
        )

        (
            self.extra_gate_pairs,
            initial_extra_thetas,
        ) = create_near_identity_extra_thetas(
            n_qubits=self.n_qubits,
            layers=EXTRA_GATE_LAYERS,
            scale=EXTRA_GATE_SCALE,
            topology=EXTRA_GATE_TOPOLOGY,
        )

        if USE_NEAR_IDENTITY_EXTENSION:
            self.extra_thetas = nn.Parameter(
                initial_extra_thetas
            )
        else:
            self.register_buffer(
                "extra_thetas",
                torch.empty(0, dtype=torch.float64),
            )

        self.quantum_device = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=TRAINING_SHOTS,
        )

        self.qnode = qml.QNode(
            self._circuit,
            self.quantum_device,
            interface="torch",
            diff_method="backprop",
        )

    def set_evaluation_shots(self, shots: int):
        self.quantum_device = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=shots,
        )

        self.qnode = qml.QNode(
            self._circuit,
            self.quantum_device,
            interface="torch",
            diff_method=None,
        )

    def encode_image(self, image):
        pixels = image.reshape(-1).to(torch.float64)

        if pixels.numel() != self.num_sites:
            raise ValueError(
                f"Expected {self.num_sites} pixels, "
                f"got {pixels.numel()}."
            )

        for site, wire in enumerate(self.geometry.image_wires):
            qml.RY(
                torch.pi * pixels[site],
                wires=wire,
            )

    def apply_variational_gates(
        self,
        first_thetas,
        middle_thetas,
        last_thetas,
    ):
        first_gate = efficient_circuit_matrix(
            self.bond_qubits,
            first_thetas,
        )

        qml.QubitUnitary(
            first_gate.to(torch.complex128),
            wires=self.geometry.first_wires,
        )

        for site in range(1, self.num_sites - 1):
            middle_gate = efficient_circuit_matrix(
                self.bond_qubits + 1,
                middle_thetas[site - 1],
            )

            qml.QubitUnitary(
                middle_gate.to(torch.complex128),
                wires=self.geometry.middle_wires[site - 1],
            )

        last_gate = efficient_circuit_matrix(
            self.bond_qubits + 1,
            last_thetas,
        )

        qml.QubitUnitary(
            last_gate.to(torch.complex128),
            wires=self.geometry.last_wires,
        )

    def apply_extra_gates(
        self,
        extra_thetas,
    ):
        if not USE_NEAR_IDENTITY_EXTENSION:
            return

        for layer in range(EXTRA_GATE_LAYERS):
            for gate_index, pair in enumerate(self.extra_gate_pairs):
                gate = rotation_matrix_nd(
                    4,
                    extra_thetas[layer, gate_index],
                )

                qml.QubitUnitary(
                    gate.to(torch.complex128),
                    wires=list(pair),
                )

    def readout_wire(self):
        if READOUT_WIRE_MODE == "mps_output":
            return self.geometry.last_wires[0]

        if READOUT_WIRE_MODE == "first":
            return 0

        if READOUT_WIRE_MODE == "last":
            return self.n_qubits - 1

        raise ValueError(
            "READOUT_WIRE_MODE must be "
            "'mps_output', 'first', or 'last'."
        )

    def _circuit(
        self,
        image,
        first_thetas,
        middle_thetas,
        last_thetas,
        extra_thetas,
    ):
        self.encode_image(image)

        self.apply_variational_gates(
            first_thetas,
            middle_thetas,
            last_thetas,
        )

        self.apply_extra_gates(extra_thetas)

        return qml.probs(
            wires=self.readout_wire()
        )

    def probability_output(self, image):
        return self.qnode(
            image,
            self.first_thetas,
            self.middle_thetas,
            self.last_thetas,
            self.extra_thetas,
        )

    def forward(self, images):
        return torch.stack(
            [
                self.probability_output(image)
                for image in images
            ]
        )


# ============================================================
# VQC training
# ============================================================

@torch.no_grad()
def evaluate_vqc(
    model,
    data_loader,
):
    model.eval()

    criterion = nn.NLLLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(torch.float64)
        labels = labels.to(torch.long)

        probabilities = model(images)

        log_probabilities = torch.log(
            probabilities.clamp_min(EPSILON)
        )

        loss = criterion(
            log_probabilities,
            labels,
        )

        predictions = probabilities.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def train_vqc_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.NLLLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    best_test_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for images, labels in train_loader:
            images = images.to(torch.float64)
            labels = labels.to(torch.long)

            optimizer.zero_grad()

            probabilities = model(images)

            log_probabilities = torch.log(
                probabilities.clamp_min(EPSILON)
            )

            loss = criterion(
                log_probabilities,
                labels,
            )

            loss.backward()
            optimizer.step()

            predictions = (
                probabilities.detach().argmax(dim=1)
            )

            batch_size = labels.size(0)

            epoch_loss += loss.item() * batch_size
            epoch_correct += (
                predictions == labels
            ).sum().item()
            epoch_examples += batch_size

        train_loss = epoch_loss / epoch_examples
        train_accuracy = epoch_correct / epoch_examples

        test_loss, test_accuracy = evaluate_vqc(
            model,
            test_loader,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[VQC] Epoch {epoch:03d}/{num_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"train acc: {train_accuracy:.2%} | "
            f"test loss: {test_loss:.6f} | "
            f"test acc: {test_accuracy:.2%}"
        )

    model.load_state_dict(best_state)

    print(
        f"\n[VQC] Best test accuracy: "
        f"{best_test_accuracy:.4f}\n"
    )

    return model, history


# ============================================================
# Experiment
# ============================================================

def mean_and_std(values):
    values = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return (
        values.mean().item(),
        values.std(unbiased=False).item(),
    )


def build_vqc_from_initialization(
    initialization: str,
    mps_model,
):
    initialization = initialization.lower()

    if initialization == "pretrained":
        first_thetas = mps_model.first_thetas.detach().clone()
        middle_thetas = mps_model.middle_thetas.detach().clone()
        last_thetas = mps_model.last_thetas.detach().clone()

    elif initialization == "non_pretrained":
        (
            first_thetas,
            middle_thetas,
            last_thetas,
        ) = create_random_vqc_thetas(
            image_size=IMAGE_SIZE,
            bond_dim=BOND_DIM,
            scale=VQC_RANDOM_SCALE,
        )

    else:
        raise ValueError(
            "initialization must be 'pretrained' "
            "or 'non_pretrained'. "
            f"Got {initialization!r}."
        )

    return MPSInitializedDynamicVQC(
        image_size=IMAGE_SIZE,
        bond_dim=BOND_DIM,
        first_thetas=first_thetas,
        middle_thetas=middle_thetas,
        last_thetas=last_thetas,
    )


def run_one_seed(seed: int):
    set_seed(seed)

    num_sites = IMAGE_SIZE * IMAGE_SIZE

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        mps_train_loader,
        mps_test_loader,
        vqc_train_loader,
        vqc_test_loader,
    ) = make_loaders(
        IMAGE_SIZE,
        seed=seed,
    )

    print("\n" + "#" * 78)
    print(f"SEED {seed}")
    print("#" * 78)

    mps_model = MPS1(
        num_sites=num_sites,
        bond_dim=BOND_DIM,
        physical_dim=PHYSICAL_DIM,
        output_dim=OUTPUT_DIM,
    )

    mps_model, _ = train_mps_model(
        model=mps_model,
        train_loader=mps_train_loader,
        test_loader=mps_test_loader,
        num_epochs=MPS_EPOCHS,
        learning_rate=MPS_LEARNING_RATE,
        device=device,
    )

    mps_model = mps_model.cpu()

    seed_results = {"seed": seed}

    for initialization in (
        "pretrained",
        "non_pretrained",
    ):
        branch_offset = (
            0
            if initialization == "pretrained"
            else 10_000
        )

        set_seed(seed + branch_offset)

        print("=" * 78)
        print(f"VQC branch: {initialization}")
        print("=" * 78)

        vqc_model = build_vqc_from_initialization(
            initialization=initialization,
            mps_model=mps_model,
        )

        (
            initial_train_loss,
            initial_train_accuracy,
        ) = evaluate_vqc(
            vqc_model,
            vqc_train_loader,
        )

        (
            initial_test_loss,
            initial_test_accuracy,
        ) = evaluate_vqc(
            vqc_model,
            vqc_test_loader,
        )

        print(
            f"[{initialization}] "
            f"initial VQC train loss="
            f"{initial_train_loss:.6f} | "
            f"initial train accuracy="
            f"{initial_train_accuracy:.2%}"
        )

        print(
            f"[{initialization}] "
            f"initial VQC test loss="
            f"{initial_test_loss:.6f} | "
            f"initial test accuracy="
            f"{initial_test_accuracy:.2%}"
        )

        vqc_model, vqc_history = train_vqc_model(
            model=vqc_model,
            train_loader=vqc_train_loader,
            test_loader=vqc_test_loader,
            num_epochs=VQC_EPOCHS,
            learning_rate=VQC_LEARNING_RATE,
        )

        vqc_model.set_evaluation_shots(EVALUATION_SHOTS)

        final_loss, final_accuracy = evaluate_vqc(
            vqc_model,
            vqc_test_loader,
        )

        seed_results[initialization] = {
            "initial_train_loss": initial_train_loss,
            "initial_train_accuracy": initial_train_accuracy,
            "initial_test_loss": initial_test_loss,
            "initial_test_accuracy": initial_test_accuracy,
            "final_test_loss": final_loss,
            "final_test_accuracy": final_accuracy,
            "initial_loss": initial_test_loss,
            "initial_accuracy": initial_test_accuracy,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "history": vqc_history,
        }

        print(
            f"[{initialization}] "
            f"initial train loss="
            f"{initial_train_loss:.6f}, "
            f"initial train acc="
            f"{initial_train_accuracy:.4f}, "
            f"initial test loss="
            f"{initial_test_loss:.6f}, "
            f"initial test acc="
            f"{initial_test_accuracy:.4f}, "
            f"final test loss="
            f"{final_loss:.6f}, "
            f"final test acc="
            f"{final_accuracy:.4f}"
        )

    return seed_results


def print_multi_seed_summary(all_results):
    print("\n" + "=" * 78)
    print("MULTI-SEED SUMMARY")
    print("=" * 78)

    print(
        f"{'seed':>6} | "
        f"{'pretrained init':>15} | "
        f"{'pretrained final':>16} | "
        f"{'non-pre init':>12} | "
        f"{'non-pre final':>13}"
    )

    print("-" * 78)

    for result in all_results:
        pre = result["pretrained"]
        non = result["non_pretrained"]

        print(
            f"{result['seed']:>6} | "
            f"{pre['initial_accuracy']:>15.4f} | "
            f"{pre['final_accuracy']:>16.4f} | "
            f"{non['initial_accuracy']:>12.4f} | "
            f"{non['final_accuracy']:>13.4f}"
        )

    print("-" * 78)

    for initialization in (
        "pretrained",
        "non_pretrained",
    ):
        initial_values = [
            result[initialization]["initial_accuracy"]
            for result in all_results
        ]

        final_values = [
            result[initialization]["final_accuracy"]
            for result in all_results
        ]

        initial_mean, initial_std = mean_and_std(initial_values)
        final_mean, final_std = mean_and_std(final_values)

        print(
            f"{initialization:>16}: "
            f"initial={initial_mean:.4f} "
            f"+/- {initial_std:.4f}, "
            f"final={final_mean:.4f} "
            f"+/- {final_std:.4f}"
        )

    paired_improvements = [
        result["pretrained"]["final_accuracy"]
        - result["non_pretrained"]["final_accuracy"]
        for result in all_results
    ]

    difference_mean, difference_std = mean_and_std(
        paired_improvements
    )

    print(
        "pretrained - non-pretrained final accuracy: "
        f"{difference_mean:+.4f} "
        f"+/- {difference_std:.4f}"
    )

    print("=" * 78)


def main():
    if PHYSICAL_DIM != 2:
        raise ValueError(
            "This direct qubit encoding assumes "
            "PHYSICAL_DIM = 2."
        )

    if OUTPUT_DIM != 2:
        raise ValueError(
            "This binary classifier assumes "
            "OUTPUT_DIM = 2."
        )

    print("=" * 78)
    print("Pretrained vs non-pretrained VQC comparison")
    print("=" * 78)
    print(f"Seeds:             {SEEDS}")
    print(f"Image size:        {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Bond dimension:    {BOND_DIM}")
    print(f"MPS epochs:        {MPS_EPOCHS}")
    print(f"VQC epochs:        {VQC_EPOCHS}")
    print(f"Final eval shots:  {EVALUATION_SHOTS}")
    print("=" * 78)

    all_results = [
        run_one_seed(seed)
        for seed in SEEDS
    ]

    print_multi_seed_summary(all_results)


if __name__ == "__main__":
    main()
