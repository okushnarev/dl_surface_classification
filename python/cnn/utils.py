from models.cnn import CNNLayerConfig, MLPLayerConfig


def configs_to_dict(cnn_config: list[CNNLayerConfig], mlp_config: list[MLPLayerConfig]):
    return dict(
        cnn_config=[item.model_dump() for item in cnn_config],
        mlp_config=[item.model_dump() for item in mlp_config],
    )


def dict_to_configs(d: dict):
    cnn_config = [CNNLayerConfig(**item) for item in d['cnn_config']]
    mlp_config = [MLPLayerConfig(**item) for item in d['mlp_config']]

    return cnn_config, mlp_config
