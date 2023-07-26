import torch
from .optimizer import load_optim
from .loss_function import load_loss_function
from .metric import metric


class Trainer:
    def __init__(self, setting, model, dataloader, logger):
        self.model = model
        self.train_dataloader = dataloader["train"]
        self.valid_dataloader = dataloader["valid"]

        self.optimizer = load_optim(model, setting["optim"])
        self.loss_function = load_loss_function(setting["loss_fn"])
        self.epoch = setting["epoch"]

        self.logger = logger

        self.model_hyperparams = setting[setting["model_name"].lower()]

        self.device = setting["device"]

        # Best
        self.best_auc = -1
        self.best_epoch = -1

        self.save_model_path = setting["path"]["save_model"]
        self.save_model_name = setting["file_name"]["save_model"]

    def train(self):
        for epoch in range(self.epoch):
            self.model.train()
            tr_loss, tr_auc, tr_acc = self._train_epoch()
            self.monitering(epoch, "Train", tr_loss, tr_auc, tr_acc)

            self.model.eval()
            va_loss, va_auc, va_acc, va_ratio, va_rmse = self._valid_epoch()
            self.monitering(
                epoch, "Validation", va_loss, va_auc, va_acc, va_ratio, va_rmse
            )

            if self.best_auc < va_auc:
                self.save_model(epoch, va_auc)
                self.logger.info(
                    f"Update Best Model epoch:{self.best_epoch}, auc:{self.best_auc} => epoch:{epoch+1}, auc:{va_auc}"
                )
                self.best_auc, self.best_epoch = va_auc, epoch

        self.logger.info(f"Save File Name : {self.save_model_name}")

    def _train_epoch(self):
        total_preds = []
        total_targets = []
        losses = []

        for edges, labels in self.train_dataloader:
            edges = edges.T
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(edges)
            loss = self.loss_function(preds, labels.to(preds.dtype))
            probs = self.model.predict_link(edge_index=edges, prob=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_preds.append(probs.detach())
            total_targets.append(labels.detach())
            losses.append(loss)

        total_targets = torch.concat(total_targets).cpu().numpy()
        total_preds = torch.concat(total_preds).cpu().numpy()

        auc, acc = metric(total_targets, total_preds)
        return sum(losses) / len(losses), auc, acc

    def _valid_epoch(self):
        with torch.no_grad():
            total_preds = []
            total_targets = []
            losses = []

            for edges, labels in self.valid_dataloader:
                edges = edges.T
                edges = edges.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(edges)
                loss = self.loss_function(preds, labels.to(preds.dtype))
                probs = self.model.predict_link(edge_index=edges, prob=True)

                total_preds.append(probs.detach())
                total_targets.append(labels.detach())
                losses.append(loss)

            total_targets = torch.concat(total_targets).cpu().numpy()
            total_preds = torch.concat(total_preds).cpu().numpy()

            auc, acc, ratio, rmse = metric(total_targets, total_preds, is_valid=True)

        return sum(losses) / len(losses), auc, acc, ratio, rmse

    def monitering(self, epoch, mode, loss, auc, acc, ratio=0, rmse=0):
        if mode == "Train":
            self.logger.info(
                f"epoch : {epoch+1} / {self.epoch} \n \t\t\t [{mode}]  | AUC: {auc:.3f} | Loss: {loss:.3f} | ACC: {acc:.3f} |"
            )
        else:
            self.logger.info(
                f"epoch : {epoch+1} / {self.epoch} \n \t\t\t [{mode}]  | AUC: {auc:.3f} | Loss: {loss:.3f} | ACC: {acc:.3f} | TN/TN+FN: {ratio:.3f} | RMSE(no_sol) : {rmse:.3f} |"
            )

    def save_model(self, epoch, best_auc):
        import os

        path = os.path.join(self.save_model_path, self.save_model_name)
        torch.save(
            {
                "epoch": epoch,
                "best_auc": best_auc,
                "model_hyperparams": self.model_hyperparams,
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )


class Tester:
    def __init__(self, setting, model, dataloader, data):
        self.test_dataloader = dataloader["test"]
        self.submission_df = data["test"]
        self.model = model
        self.device = setting["device"]
        self.save_path = setting["path"]["submission"]
        self.save_file_name = setting["file_name"]["submission"]

    def test(self):
        self.model.eval()
        with torch.no_grad():
            total_preds = []
            for edges, _ in self.test_dataloader:
                edges = edges.T.to(self.device)
                probs = self.model.predict_link(edge_index=edges, prob=True)
                total_preds.append(probs.detach())
            total_preds = torch.concat(total_preds).cpu().numpy()
            self.submission_df["answer"] = total_preds

    def save_submission(self):
        import os

        self.submission_df.to_csv(
            os.path.join(self.save_path, self.save_file_name), index=False
        )
