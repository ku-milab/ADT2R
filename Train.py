import torch
import numpy as np
from tqdm import tqdm

import Load as ld
import Utils as ut
import Module as md


def train_one_epoch(model, loader, device):
    model.train()
    loss_all = 0
    for bidx, batch in enumerate(loader):
        record = dict()
        for key in batch.keys():
            record[key] = batch[key].to(device)

        loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, update=True)
        loss_all_b = loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b
        loss_all += loss_all_b.cpu().detach().numpy()

    return loss_all / len(loader)

def eval_(model, loader, device):
    model.eval()
    loss_all = 0
    preds, reals, masks, probs, rewards = [], [], [], [], []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            record = dict()
            for key in batch.keys():
                record[key]  = batch[key].to(device)

            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, update=False)

            loss_all_b = loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b
            loss_all += loss_all_b.cpu().detach().numpy()

            preds.extend(pred_b.cpu().detach().numpy())
            reals.extend(batch["action"].cpu().detach().numpy())
            masks.extend(batch["seq_mask"].cpu().detach().numpy())
            probs.extend(prob_b.cpu().detach().numpy())
            rewards.extend(record["mortality"].cpu().detach().numpy())

    acc, jaccard, recall, wis = ut.calculate_metric(np.array(reals), np.array(preds), np.array(masks), np.array(probs), np.array(rewards))

    return loss_all / len(loader), acc, jaccard, recall, wis


def run(args, device, exp_name):
    """ Load datasets """
    print("**\t", exp_name)
    print("**\t Load dataset")

    train_loader, valid_loader, test_loader = ld.load_fold(args)

    model = md.ADT2R(args.state_dim, args.action_dim, args.h_dim, args.n_heads, args.drop_p, args.max_timestep, device, args.lr, args.w_decay, args.lr_decay, args.lr_step, args.lam_actor, args.lam_critic, args.lam_reg, args.gamma, args.tau).to(device)
    scheduler = model.scheduler

    for ep in tqdm(range(args.total_epoch)):

        tr_loss = train_one_epoch(model, train_loader, device)
        scheduler.step()

        vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis = eval_(model, valid_loader, device)
        ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis = eval_(model, test_loader, device)

