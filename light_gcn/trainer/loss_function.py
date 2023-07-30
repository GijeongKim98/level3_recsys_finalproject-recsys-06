from torch.nn import BCEWithLogitsLoss


def load_loss_function(loss_fn_name):
    if loss_fn_name == "bceloss":
        loss_fn = BCEWithLogitsLoss()
    else:
        raise NameError(f"NameError Not Found loss function : {loss_fn_name}")
    return loss_fn
