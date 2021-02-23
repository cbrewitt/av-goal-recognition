from torch.utils.data import Dataset

from core.generate_dataset_split import load_dataset_splits

class GRITDataset(Dataset):
    def __init__(self, scenario_name, split_type="train"):
        self.scenario_name = scenario_name
        self.split_type = split_type
        self.dataset_split = load_dataset_splits()[self.scenario_name][self.split_type]

        self.dataset = None
        self.labels = None
        self.lengths = None

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index], self.lengths[index]