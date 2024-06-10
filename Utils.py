import os, glob, json
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, classification_report


def save_configuration(path, name, arg):
    with open(path + name, "w") as f:
        json.dump(arg.__dict__, f, indent=2)
    print("Save configuration.")

def read_file_name(query, pool_path):
    answer = glob.glob(pool_path + query)
    return answer

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def delete_dir(path):
    if os.path.isfile(path):
        os.remove(path)

def calculate_wis(probs_list, trues_list, rewards_list, masks_list, gamma=0.99):
    """ Reference: https://github.com/yinchangchang/DAC/blob/a36cef7b94464d07eca4f317e3c235aca7fcdd81/tools/py_op.py#L130 """

    rho_list = []
    for probs, rewards, masks, reals in zip(probs_list, rewards_list, masks_list, trues_list):
        assert len(probs) == len(rewards) == len(masks) == len(reals)
        rho = []
        for prob, reward, mask, real in zip(probs, rewards, masks, reals):  # T
            if mask == 0:
                break
            prob = min(1, max(0, prob[real]))
            if len(rho) == 0:
                rho.append(prob)
            else:
                rho.append(prob * rho[-1])
        rho_list.append(rho)

    max_step = max([len(rho) for rho in rho_list])

    w_list = []
    for i in range(max_step):
        w_h = []
        for rho in rho_list:
            if len(rho) > i:
                w_h.append(rho[i])

        w_list.append(np.mean(w_h))

    v_list = []
    for rho, rs, ms in zip(rho_list, rewards_list, masks_list):
        h = len(rho)
        if h <= 1:
            continue

        assert rho[h - 1] <= 1
        assert w_list[h - 1] <= 1
        assert len(rs) > 0
        assert rs[h - 1] != 0
        v_wis = rho[h - 1] / (w_list[h - 1] + 1e-6) * rs[h - 1] * np.power(gamma, len(rho) - 1)
        v_list.append(v_wis)

    return np.mean(v_list)


def calculate_metric(reals, preds, masks, probs, rewards):

    a_true = reals.flatten()
    a_pred = preds.flatten()
    mask = masks.flatten()

    mask_idx = np.nonzero(mask)
    a_true = a_true[mask_idx[0]]
    a_pred = a_pred[mask_idx[0]]

    acc = accuracy_score(a_true, a_pred)

    jaccard = jaccard_score(a_true, a_pred, average="micro")

    report = classification_report(a_true, a_pred, output_dict=True)
    recall = report["macro avg"]["recall"]

    wis = calculate_wis(probs, reals, rewards, masks)


    return acc, jaccard, recall, wis