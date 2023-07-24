import torch
from torch.utils.data import Dataset


class GraphBaseDataset(Dataset):
    def __init__(self, df, num_nodes, node2idx):
        self.num_nodes = num_nodes
        self.edges, self.labels = [], []
        for user_id, problem_id, answer_code in zip(
            df["user_id"], df["problem_id"], df["answer_code"]
        ):
            self.edges.append([node2idx[user_id], node2idx[problem_id]])
            self.labels.append(answer_code)

        self.edges = torch.LongTensor(self.edges)
        # print(self.edges)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, x):
        return self.edges[x], self.labels[x]

    def __len__(self):
        return len(self.labels)
