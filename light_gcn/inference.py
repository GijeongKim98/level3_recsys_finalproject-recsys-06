# module
from utils.util import init
from preprocess import preprocess, split
from dataset import create_dataset, create_dataloader
from model.model import load_model
from trainer.trainer import Tester


def main():
    # load data & setting
    setting, data, logger = init()
    logger.info("Succes Initializing setting, data, logger")

    # preprocessing
    data = preprocess(setting, data, logger)

    # dataset
    dataset = create_dataset(setting, data, is_train=False)
    logger.info("Succes Creating Dataset")

    # dataloader
    dataloader = create_dataloader(setting, dataset)
    logger.info("Succes Creating Dataloader")

    # create_model
    model = load_model(setting, data["num_nodes"], logger, is_train=False)
    logger.info(f"Succes Load Model : {setting['model_name']}")

    # trainer
    tester = Tester(setting, model, dataloader, data)
    logger.info(f"Succes Initialize Tester")

    # Model Train
    logger.info(f"Strat Model Test")
    tester.test()

    logger.info(f"Save Submissionfile.")
    tester.save_submission()


if __name__ == "__main__":
    main()
