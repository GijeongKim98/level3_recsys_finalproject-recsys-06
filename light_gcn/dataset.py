from datasets.graph_base_dataset import GraphBaseDataset
from torch.utils.data import DataLoader


def create_dataset(setting, data, is_train=True):
    model_name = setting["model_name"].lower()
    dataset = dict()
    if not is_train:
        if model_name == "lgcn":
            dataset["test"] = GraphBaseDataset(
                data["test"], data["num_nodes"], data["node2idx"]
            )
            return dataset
    if model_name == "lgcn":
        dataset["train"] = GraphBaseDataset(
            data["train"], data["num_nodes"], data["node2idx"]
        )
        dataset["valid"] = GraphBaseDataset(
            data["valid"], data["num_nodes"], data["node2idx"]
        )
    else:
        raise KeyError(f"{model_name} Not Found")
    return dataset


def create_dataloader(setting: dict, dataset: dict, is_train=True):
    model_name = setting["model_name"].lower()
    dataloader = dict()
    for key in dataset.keys():
        dataloader[key] = DataLoader(
            dataset=dataset[key], batch_size=setting["batch_size"], shuffle=False
        )
    return dataloader
