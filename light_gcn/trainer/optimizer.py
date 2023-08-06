from torch.optim import Adam, SGD, RMSprop, Adadelta


def load_optim(model, optim_config):
    optim_name = optim_config["name"]
    lr = optim_config["learning_rate"]
    weight_decay = optim_config["weight_decay"]
    if optim_name == "adam":
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        optim = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "rmsprop":
        optim = RMSprob(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "adadelta":
        optim = Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NameError(f"Not Found Optimizer : {optim_name}")
    optim.zero_grad()
    return optim
