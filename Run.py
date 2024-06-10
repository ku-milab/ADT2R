import argparse, torch, os

import Train as tr
import Utils as ut


def parse_args():
    parser = argparse.ArgumentParser("ADTR")

    parser.add_argument('--state_dim', type=int, default=38)
    parser.add_argument('--action_dim', type=int, default=25)
    parser.add_argument("--h_dim", type=int, default=60)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--drop_p", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay")
    parser.add_argument("--lr_step", type=int, default=2, help="Learning rate decay stepsize")
    parser.add_argument("--w_decay", default=0.0001, type=float, help="Weight decay (lambda)")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--total_epoch", type=int, default=100, help="# of epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--lam_reg", type=float)
    parser.add_argument("--lam_critic", type=float)
    parser.add_argument("--lam_actor", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tau", type=float)

    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0, help="5-fold cross validation")
    parser.add_argument("--max_timestep", default=20, type=int)

    parser.add_argument("--config_path", type=str, default="./Configuration/")
    parser.add_argument("--data_path", type=str, default="./Data/")

    return parser.parse_args()

if "__main__" == __name__:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_name = f"ADTR_Fold{args.fold}"

    # Save configuration
    ut.save_configuration(args.config_path, exp_name+"_Configuration.txt", args)
    tr.run(args, device, exp_name)
