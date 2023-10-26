from torch.utils.data.dataset import Dataset


class DictWrapperDataset(Dataset):
    def __init__(self, dataset, key_list):
        self.dataset = dataset
        self.key_list = key_list

    def __getitem__(self, index):
        item = self.dataset[index]

        output = {key: value for key, value in zip(self.key_list, item)}

        output["indices"] = index

        return output

    def __len__(self):
        return len(self.dataset)
