import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


""" Indices of the interested EHR variables """
demo_idx = np.array([])
vital_idx = np.array([])
lab_idx = np.array([])
sofa_idx = np.array([])
action_idx = np.array([])

def load_fold(args):

    trainset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Train", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)
    validset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Valid", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)
    testset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Test", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)

    trainloader = DataLoader(trainset, args.bs, shuffle=True)
    validloader = DataLoader(validset, args.bs, shuffle=False)
    testloader = DataLoader(testset, args.bs, shuffle=False)

    return trainloader, validloader, testloader


class CustomDataset(Dataset):
    def __init__(self, data_path, interval=4, fold=1, mode="Train", missing_rate=0, max_length=20, n_class=25):
        self.missing_rate = missing_rate
        self.interval = interval
        self.fold = fold
        self.mode = mode
        self.T = max_length
        self.n_class = n_class

        if self.mode in ["Train", "train"]:
            self.fpath = data_path + f"Fold{self.fold}_Train.csv"
        elif self.mode in ["Valid", "valid"]:
            self.fpath = data_path + f"Fold{self.fold}_Valid.csv"
        elif self.mode in ["Test", "test"]:
            self.fpath = data_path + f"Fold{self.fold}_Test.csv"
        else:
            raise KeyError("Unknown mode. You should select one among train, valid, and test.")

        self.df = pd.read_csv(self.fpath)
        self.head = self.df.columns
        self.pindices = self.df["traj"].unique()

    def parse_delta(self, mask, direction, time):
        """
        :param mask: masking vectors
        :param direction: forward or backward
        :param time: Real time interval ("s" in BRITS paper)
        :return:
        """

        if direction == "backward":
            mask = mask[::-1]

        [T, D] = mask.shape
        deltas = []
        for t in range(T):
            if t == 0:
                deltas.append(np.zeros(D))
            else: # t>0
                deltas.append(np.ones(D) * time[t] - np.ones(D) * time[t-1] + (1-mask[t-1]) * deltas[-1])

        return np.array(deltas)

    def calculate_rtg(self, rewards):
        """
        :param rewards: ndarray
        :return:
        """
        return np.flip(np.cumsum(np.flip(rewards)))

    def get_data(self, idx):
        condition = self.df.traj == idx
        vitals = self.df[condition].iloc[:, vital_idx].values #[T, 8]
        labs = self.df[condition].iloc[:, lab_idx].values #[T, 22]
        sofas = self.df[condition].iloc[:, sofa_idx].values #[T, 7]
        actions = self.df[condition].iloc[:, action_idx[0]].values #[T,]
        sequence = self.df[condition].iloc[:, 1].values #[T,]
        mortality = self.df[condition].iloc[:, -1].values
        reward = self.df[condition].iloc[:, -1].values * 15
        demo = self.df[condition].iloc[:, demo_idx].values

        mortality_last = np.array([mortality[-1]])

        # Get SOFA scores
        sofa_res = sofas[:, 0]
        sofa_coa = sofas[:, 1]
        sofa_liv = sofas[:, 2]
        sofa_car = sofas[:, 3]
        sofa_cns = sofas[:, 4]
        sofa_ren = sofas[:, 5]
        sofa_all = sofas[:, -1]

        mask_sofa_res = ~np.isnan(sofa_res)
        mask_sofa_coa = ~np.isnan(sofa_coa)
        mask_sofa_liv = ~np.isnan(sofa_liv)
        mask_sofa_car = ~np.isnan(sofa_car)
        mask_sofa_cns = ~np.isnan(sofa_cns)
        mask_sofa_ren = ~np.isnan(sofa_ren)
        mask_sofa_all = ~np.isnan(sofas[:,:-1])


        # Get EHR variables
        data = np.concatenate((vitals, labs), axis=-1)  # [sequence, variables]
        mask_data_real = ~ np.isnan(data)  # 1: observed, 0: missing
        delta = self.parse_delta(mask_data_real, direction="forward", time=sequence)
        delta_sofa = self.parse_delta(mask_sofa_all, direction="forward", time=sequence)
        seq_mask = np.ones(shape=(sequence.shape[0]))

        # Pad with 0 in data to construct a mini-batch
        if sequence.shape[0] < self.T:
            rtg = self.calculate_rtg(np.delete(sofa_all, 0) - np.delete(sofa_all, -1))
            rtg = np.concatenate((rtg, mortality_last*15))

            demo = np.concatenate((demo, np.zeros((int(self.T - sequence.shape[0]), demo.shape[1]))), axis=0)  # [T, n_variable]
            data = np.concatenate((data, np.zeros((int(self.T - sequence.shape[0]), data.shape[1]))),
                                  axis=0)  # [T, n_variable]
            mask_data_real = np.concatenate((mask_data_real, np.zeros((int(self.T - sequence.shape[0]), mask_data_real.shape[1]))), axis=0)  # [T, n_variable]
            mask_sofa_all = np.concatenate((mask_sofa_all, np.zeros((int(self.T - sequence.shape[0]), mask_sofa_all.shape[1]))), axis=0)

            actions = np.concatenate((actions, np.zeros((int(self.T - sequence.shape[0])))), axis=0)
            mortality = np.concatenate((mortality, np.zeros((int(self.T - sequence.shape[0])))), axis=0)

            reward = np.concatenate((reward, np.zeros((int(self.T - sequence.shape[0])))), axis=0)

            mask_sofa_res = np.concatenate((mask_sofa_res, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_coa = np.concatenate((mask_sofa_coa, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_liv = np.concatenate((mask_sofa_liv, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_car = np.concatenate((mask_sofa_car, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_cns = np.concatenate((mask_sofa_cns, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_ren = np.concatenate((mask_sofa_ren, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]

            sofa_res = np.concatenate((sofa_res, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_coa = np.concatenate((sofa_coa, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_liv = np.concatenate((sofa_liv, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_car = np.concatenate((sofa_car, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_cns = np.concatenate((sofa_cns, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_ren = np.concatenate((sofa_ren, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_all = np.concatenate((sofa_all, np.zeros((int(self.T - sequence.shape[0])))), axis=0) #[T]

            seq_mask = np.concatenate((seq_mask, np.zeros((int(self.T - sequence.shape[0])))), axis=0)
            sequence = np.concatenate((sequence, np.nan * np.ones((int(self.T - sequence.shape[0])))), axis=0)

        else:
            demo = demo[:self.T]
            data = data[:self.T]
            mask_data_real = mask_data_real[:self.T]
            delta_real = delta[:self.T]
            delta_sofa = delta_sofa[:self.T]
            actions = actions[:self.T]
            mask_sofa_res = mask_sofa_res[:self.T]
            mask_sofa_coa = mask_sofa_coa[:self.T]
            mask_sofa_liv = mask_sofa_liv[:self.T]
            mask_sofa_car = mask_sofa_car[:self.T]
            mask_sofa_cns = mask_sofa_cns[:self.T]
            mask_sofa_ren = mask_sofa_ren[:self.T]
            mask_sofa_all = mask_sofa_all[:self.T]

            sofa_all = sofa_all[:self.T]
            sofa_res = sofa_res[:self.T]
            sofa_coa = sofa_coa[:self.T]
            sofa_liv = sofa_liv[:self.T]
            sofa_car = sofa_car[:self.T]
            sofa_cns = sofa_cns[:self.T]
            sofa_ren = sofa_ren[:self.T]
            sequence = sequence[:self.T]
            seq_mask = seq_mask[:self.T]
            mortality = mortality[:self.T]
            reward = reward[:self.T]


        record = dict()
        record["observation"] = torch.from_numpy(np.nan_to_num(data, nan=0)).float()
        record["action"] = torch.from_numpy(actions.astype("long"))
        record["sequence"] = torch.from_numpy(np.nan_to_num(sequence, nan=0)).float()
        record["seq_mask"] = torch.from_numpy((seq_mask.astype("int32")))
        record["mask_data_real"] = torch.from_numpy(mask_data_real.astype("int32"))

        record["mask_sofa_res"] = torch.from_numpy(mask_sofa_res).float()
        record["mask_sofa_coa"] = torch.from_numpy(mask_sofa_coa).float()
        record["mask_sofa_liv"] = torch.from_numpy(mask_sofa_liv).float()
        record["mask_sofa_car"] = torch.from_numpy(mask_sofa_car).float()
        record["mask_sofa_cns"] = torch.from_numpy(mask_sofa_cns).float()
        record["mask_sofa_ren"] = torch.from_numpy(mask_sofa_ren).float()
        record["mask_sofa_all"] = torch.from_numpy(mask_sofa_all).float()

        record["sofa_res"] = torch.from_numpy(np.nan_to_num(sofa_res, nan=0).astype("long"))
        record["sofa_coa"] = torch.from_numpy(np.nan_to_num(sofa_coa, nan=0).astype("long"))
        record["sofa_liv"] = torch.from_numpy(np.nan_to_num(sofa_liv, nan=0).astype("long"))
        record["sofa_car"] = torch.from_numpy(np.nan_to_num(sofa_car, nan=0).astype("long"))
        record["sofa_cns"] = torch.from_numpy(np.nan_to_num(sofa_cns, nan=0).astype("long"))
        record["sofa_ren"] = torch.from_numpy(np.nan_to_num(sofa_ren, nan=0).astype("long"))
        record["sofa_all"] = torch.from_numpy(np.nan_to_num(sofa_all, nan=0).astype("long"))

        record["mortality"] = torch.from_numpy(mortality).float()
        record["reward"] = torch.from_numpy(reward).float()
        record["demo"] = torch.from_numpy(np.nan_to_num(demo, nan=0)).float()
        record["mortality_last"] = torch.from_numpy(mortality_last).float()


        return record

    def __getitem__(self, idx):
        return self.get_data(self.pindices[idx])

    def __len__(self):
        return len(self.pindices)  # the number of patients