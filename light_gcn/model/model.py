import torch
import torch.nn as nn
from torch_geometric.nn.models import LightGCN


def load_model(setting: dict, num_nodes, logger, is_train: bool = True) -> nn.Module:
    """Load a Model

    Args:
        setting (dict): Required Settings
        is_train (bool): if train = Create model object ,
                         else = Load trained model

    Raises:
        NameError: Occurs when the selected model does not exist

    Returns:
        nn.Module: Return model
    """
    model_name = setting["model_name"].lower()

    if model_name == "lgcn":
        model = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=setting["lgcn"]["embedding_dim"],
            num_layers=setting["lgcn"]["num_layers"],
        )
    else:
        raise NameError(f"Not Found model : {model_name}")

    if not is_train:
        model_path = get_model_path(setting)
        model_dict = torch.load(model_path)
        logger.info(f"Load Model : {model_name}, best_auc: {model_dict['best_auc']} ")
        model.load_state_dict(model_dict["model_state_dict"].to(setting["device"]))

    return model.to(setting["device"])


# utils
def get_model_path(setting):
    import os

    return os.path.join(
        setting["path"]["save_model"], setting["file_name"]["save_model"]
    )
