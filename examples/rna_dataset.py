from torch.utils import data


class DNADataset(data.Dataset):
    def __init__(self, input_ids, mask, type_ids, label):
        self.input_ids = input_ids
        self.mask = mask
        self.type_ids = type_ids
        self.label = label

    def __getitem__(self, index):
        datas = self.input_ids[index]
        masks = self.mask[index]
        type_ids = self.type_ids[index]
        labels = self.label[index]

        return datas, masks, type_ids, labels

    def __len__(self):
        return len(self.input_ids)
