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
python dense_exp.py
```

or

```bash
python efficient_exp.py
```

## About

MSc in Theoretical Physics  
The University of Edinburgh, 2026
