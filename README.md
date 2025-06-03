
# Unlearning Playground

This repository provides a modular and extensible framework to experiment with Machine Unlearning (MU) techniques. It includes implementations of several unlearning methods, evaluation metrics, and the core pipeline to run MU experiments across different datasets and tasks.

## 📁 Repository Structure

```
unlearning-playground-main/
│
├── unlearners.py               # Implemented unlearning methods
├── unlearning.py               # Core pipeline for running MU experiments
└── Utils/
    ├── evaluation_metrics.py   # Evaluation metrics: accuracy, F1, MIA, time
    └── utils.py                # Utility functions: set_seed, argument parsing
```

## 🔧 How to Use

1. **Define Your Task:**
   - Add your own model and dataset to `unlearning.py`.
   - Adapt the experiment logic based on your task (e.g., classification, regression).

2. **Run an Experiment:**
   - Use the argument parser defined in `Utils/utils.py` to set up hyperparameters and experimental settings.
   - Example:
     ```bash
     python unlearning.py --dataset <your_dataset> --model <your_model> --method <unlearning_method>
     ```

3. **Evaluate Results:**
   - Evaluation metrics such as Accuracy, F1, MIA (Membership Inference Attack), and Runtime are available in `Utils/evaluation_metrics.py`.

## 🧠 Implemented Unlearning Methods

The following methods are implemented in `unlearners.py`:

- **Fine-Tuning**
- **Successive Random Labels**
- **Neggrad**
- **Advanced Neggrad** 
- **UNSIR** 
- **SCRUB** 
- **Bad Teaching** 

## 📏 Evaluation Metrics

Defined in `Utils/evaluation_metrics.py`:
- **Accuracy** and **F1 score**: Measured on the test and retained sets.
- **MIA (Membership Inference Attack)**: Measures privacy leakage after unlearning.
- **Time**: Measures the computational efficiency of the unlearning process.

## 🧰 Utilities

Available in `Utils/utils.py`:
- `set_seed(seed)`: For reproducibility.
- `get_args()`: Argument parser for configuring experiments easily.
