from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def metric(total_targets, total_preds, is_valid=False):
    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )
    if not is_valid:
        return auc, acc
    no_solve_incorrect_count = 0
    no_solve_correct_count = 0

    no_solve_model_error = 0

    for idx, (label, predict) in enumerate(zip(total_targets, total_preds)):
        if label == 0:
            no_solve_model_error += predict**2
            if predict > 0.5:
                no_solve_incorrect_count += 1
            else:
                no_solve_correct_count += 1

    ratio = no_solve_incorrect_count / (
        no_solve_correct_count + no_solve_incorrect_count
    )
    rmse = np.sqrt(
        no_solve_model_error / (no_solve_correct_count + no_solve_incorrect_count)
    )

    return auc, acc, ratio, rmse
