from pathlib import Path


class ProjectPaths:
    """
    Resolves paths relative to the project root to ensure consistency across scripts
    """

    # Anchor: src/utils/paths.py -> src/utils -> src -> ROOT
    _ROOT = Path(__file__).resolve().parent.parent.parent

    @classmethod
    def get_root(cls) -> Path:
        return cls._ROOT

    @classmethod
    def get_processed_data_dir(cls, dataset_scope: str) -> Path:
        """
        Returns: datasets/processed/<dataset_scope>/
        Example: datasets/processed/main/
        """
        return cls._ROOT / 'datasets' / 'processed' / dataset_scope

    @classmethod
    def get_raw_data_dir(cls) -> Path:
        """
        Returns: datasets/raw/
        """
        return cls._ROOT / 'datasets' / 'raw'

    @classmethod
    def get_feature_config_path(cls, dataset_scope: str) -> Path:
        """
        Returns: configs/datasets/<dataset_scope>/dataset_config.json
        """
        return cls._ROOT / 'configs' / 'datasets' / dataset_scope / 'dataset_config.json'

    @classmethod
    def get_params_dir(cls, model_name: str, dataset_scope: str) -> Path:
        """
        Returns: configs/model_params/<model_name>/<dataset_scope>/
        Example: configs/model_params/rnn/main/
        """
        return cls._ROOT / 'configs' / 'model_params' / model_name / dataset_scope

    @classmethod
    def get_experiment_config_path(cls, model_name: str, config_name: str) -> Path:
        """
        Returns: configs/experiments/<model_name>/<config_name>.yaml
        Example: configs/experiments/rnn/main.yaml
        """
        return cls._ROOT / 'configs' / 'experiments' / model_name / f'{config_name}.yaml'

    @classmethod
    def get_run_dir(cls, dataset_scope: str, exp_name: str) -> Path:
        """
        Returns: experiments/<study_group>/<exp_name>/
        Example: experiments/main/rnn_no_filter_type_3_main/
        """
        return cls._ROOT / 'experiments' / dataset_scope / exp_name

    @classmethod
    def get_evaluation_dir(cls, dataset_scope: str, subset_name: str | Path) -> Path:
        """
        Returns: results/<dataset_scope>/<subset_name>/
        Example: results/main/cv/full/ or results/main/full/
        """
        return cls._ROOT / 'results' / dataset_scope / subset_name

    @classmethod
    def get_figures_dir(cls, dataset_scope: str) -> Path:
        """
        Returns: figures/<dataset_scope>/
        Example: figures/main/
        """
        return cls._ROOT / 'figures' / dataset_scope


if __name__ == '__main__':
    print(ProjectPaths.get_root())
