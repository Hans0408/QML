# Tensor Network Pretraining in Quantum Machine Learning

Code for my MSc research project on using Matrix Product States (MPS) to pretrain Variational Quantum Circuits (VQCs).

The project compares pretrained and randomly initialized VQCs on a Bars-and-Stripes classification task.

## Files

- `dense_exp.py` - Dense SO(n) MPS-to-VQC implementation
- `efficient_exp.py` - Efficient circuit-based MPS-to-VQC implementation

## Requirements

```bash
pip install torch pennylane
```

## Run

```bash
python cleaned_script_1.py
```

or

```bash
python cleaned_script_2.py
```

## About

The MPS is trained classically and its parameters are transferred directly to the VQC to provide a better initialization.

MSc in Theoretical Physics  
The University of Edinburgh, 2026
