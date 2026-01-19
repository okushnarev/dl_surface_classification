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
    def get_dataset_config_path(cls, dataset_scope: str) -> Path:
        """
        Returns: configs/datasets/<dataset_scope>/dataset_config.json
        """
        return cls._ROOT / 'configs' / 'datasets' / dataset_scope / 'dataset_config.json'

    @classmethod
    def get_params_dir(cls, model_name: str, config_name: str) -> Path:
        """
        Returns: configs/model_params/<model_name>/<config_name>/
        Example: configs/model_params/rnn/main/
        """
        return cls._ROOT / 'configs' / 'model_params' / model_name / config_name

    @classmethod
    def get_params_path(cls, model_name: str, config_name: str, exp_name: str) -> Path:
        """
        Returns: configs/model_params/<model>/<config_name>/best_params_<exp_name>.json
        """
        # We ensure the filename is safe (replace spaces with underscores if any)
        safe_name = exp_name.replace(' ', '_')
        filename = f'best_params_{safe_name}.json'

        return cls.get_params_dir(model_name, config_name) / filename

    @classmethod
    def get_experiment_config_path(cls, model_name: str, config_name: str) -> Path:
        """
        Returns: configs/experiments/<model_name>/<config_name>.yaml
        Example: configs/experiments/rnn/main.yaml
        """
        return cls._ROOT / 'configs' / 'experiments' / model_name / f'{config_name}.yaml'

    @classmethod
    def get_run_dir(cls, config_name: str, exp_name: str) -> Path:
        """
        Returns: experiments/<config_name>/<exp_name>/
        Example: experiments/main/rnn_no_filter_type_3_main/
        """
        return cls._ROOT / 'experiments' / config_name / exp_name

    @classmethod
    def get_evaluation_dir(cls, config_name: str, subset_name: str | Path) -> Path:
        """
        Returns: results/<config_name>/<subset_name>/
        Example: results/main/cv/full/ or results/main/full/
        """
        return cls._ROOT / 'results' / config_name / subset_name

    @classmethod
    def get_figures_dir(cls, config_name: str) -> Path:
        """
        Returns: figures/<config_name>/
        Example: figures/main/
        """
        return cls._ROOT / 'figures' / config_name


if __name__ == '__main__':
    print(ProjectPaths.get_root())
