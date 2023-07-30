# module
from utils.util import init
from preprocess import preprocess, split
from dataset import create_dataset, create_dataloader
from model.model import load_model
from trainer.trainer import Trainer


def main():
    # load data & setting
    setting, data, logger = init()
    logger.info("Succes Initializing setting, data, logger")

    # preprocessing
    data = preprocess(setting, data, logger)

    # split
    data = split(setting, data, logger)

    # dataset
    dataset = create_dataset(setting, data)
    logger.info("Succes Creating Dataset")

    # dataloader
    dataloader = create_dataloader(setting, dataset)
    logger.info("Succes Creating Dataloader")

    # create_model
    model = load_model(setting, data["num_nodes"], logger)
    logger.info(f"Succes Load Model : {setting['model_name']}")

    # trainer
    trainer = Trainer(setting, model, dataloader, logger)
    logger.info(f"Succes Initialize Trainer")
    logger.info(f"setting : {setting['lgcn'].items()}")
    logger.info(f"setting : {setting['optim'].items()}")

    # Model Train
    logger.info(f"Strat Model Train")
    trainer.train()


if __name__ == "__main__":
    main()
