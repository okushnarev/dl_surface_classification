# Leveraging Energy Features for Surface Classification with Deep Learning: A Comparative Analysis Across Three Independent Datasets

___

This repo contains the source code and results for paper _Leveraging Energy Features for Surface Classification with
Deep Learning: A Comparative Analysis Across Three Independent Datasets_

## Installation

This project uses `uv` for dependency management (indicated by `uv.lock` and `pyproject.toml`).

1. Clone the repository.
2. Install `uv` if not already installed.
3. Sync the environment and install dependencies:

```bash
uv sync
```

## Project Structure

The repository is organized to separate configurations, execution scripts, core source code, and artifacts (data,
models, results).

```text
.
├── baseline_stats/  # Sklearn classification report baselines (.csv)
├── configs/         # JSON/YAML configs for datasets and experiments
├── datasets/        # Raw and processed datasets (train/test/proxy splits)
├── experiments/     # Model weights (.pt), scalers, and training logs
├── figures/         # Generated plots and visualizations
├── results/         # Evaluation outputs (.csv predictions, .json hashes)
├── scripts/         # Executable entry points for the user workflow
├── src/             # Core modules (data logic, engine, models, utils)
└── tables/          # Reports in table format (.csv, .xlsx, etc.)
```

## Models

The repository includes implementations of four modern deep learning architectures designed for processing sequential
data. Architectures are defined in `src/models/nets/`

- **Recurrent Neural Network**: Utilizes GRU cells combined with a funnel-like MLP encoder and linear projection
  layer.
- **Convolutional Neural Network**: Utilizes 1D convolutional blocks and max-pooling layers, mapping flattened
  outputs to an MLP classifier.
- **Encoder-only Transformer**: Employs an MLP encoder, positional encoding, transformer layers, and a prepended
  learnable classification token for global feature aggregation.
- **Mamba**: Integrates a Mamba2 block with an expanding MLP encoder and a contracting MLP classifier.


## Configuration System

The framework requires two types of configuration files to operate:

#### 1. Dataset Configuration (`configs/datasets/<dataset_name>.json`)

Defines the metadata and feature subsets for a specific dataset.

* **Metadata**: Specifies target columns, grouping columns (to distinguish concatenated experiments), and visualization
  colors for specific surface classes.
* **Features**: Defines various sets of input features (e.g., IMU-only, Energy-only, combined) under different keys (
  e.g., `set_4`, `set_5`).

#### 2. Experiment Configuration (`configs/experiments/<model_name>/<config_name>.yaml`)

Defines the optimization and training parameters for specific runs.

* Declares a dataset that is used in all experiments for certain config
* Defines optimization and train parameters
* Lists individual experiments. Experiment's name is used for automatic naming of checkpoints, results and other related
  info.

## Usage

All user interactions are performed via the `scripts/` directory.

Typical workflow is:

#### 1. Put raw dataset files into `datasets/raw/` directory

#### 2. Write dataset config into `configs/datasets/` as `.json` file

Use existing configs for reference. State `group_cols`, `info_cols`, and `target_col` in `metadata` section, and feature
sets in `features` section. `features` expect separation by filter type. If your features are not separated by filter type simply  add `no_filter` subsection.

#### 3. Run [scripts/prepare_raw_data.py](scripts/prepare_raw_data.py)

Example:

```bash
uv run scripts/prepare_raw_data.py --dataset boreal_imu --full_ds_name boreal_imu --chunk_size 100 --proxy_size 0.18
```

#### 4. Write experiment config for specific net into `configs/experiments/<net>/` as `.yaml` file

Use existing configs for reference. State `dataset` that will be used for exepriments in `defaults` section. State
`name` of experiment in `experiments` section along with other parameters of the experiments

#### 5. Run [scripts/optimize.py](scripts/optimize.py)

Example:

```bash
uv run scripts/optimize.py --config_name kedzierski_4W --nn_name cnn
```

#### 6. Run [scripts/train.py](scripts/train.py)

Example:

```bash
uv run scripts/train.py --config_name kedzierski_4W --nn_name cnn
```

#### 7. Run [scripts/evaluate.py](scripts/evaluate.py)

It is possible to use only one `config` and `dataset` here but many `nets`

Example:

```bash
uv run scripts/evaluate.py --dataset belyaev_kushnarev --config_name belyaev_kushnarev --nets cnn rnn transformer mamba --subset test --ckpt_type last --batch_size 8000
```

#### (Optional) 8. Run [aggregate_results_to_table.py](scripts/aggregate_results_to_table.py)
It is a way to analyze nets' performance

Example

```bash
uv run scripts/aggregate_results_to_table.py --nets rnn cnn transformer mamba --configs boreal --subset test --baseline_df boreal_cnn
```

