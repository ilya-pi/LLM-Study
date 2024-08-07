import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # output layer
                torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

# In order to make random number initialization reproducible
torch.manual_seed(123)

model = NeuralNetwork(50, 3)

# Print summary of the created model
print(model)

# Total amount of trainable parameters of this model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(f"Weight parameters matrix for layer 0: {model.layers[0].weight} \n with shape {model.layers[0].weight.shape} and a bias vector {model.layers[0].bias}")

# Forward pass of the model
X = torch.rand((1, 50))
out = model(X)
print("Forward pass:")
print(out)

# Forward pass of the model without preserving gradients (for inference for example)
with torch.no_grad():
    out = model(X)
    print("Forward pass without keeing gradients")
    print(out)

# With explicit softmax
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
    print("Model outputs with class-memeber probabilities")
    print(out)


X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
    ])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
    ])
y_test = torch.tensor([0, 1])

from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(f"We have {len(train_ds)} five rows in the dataset")

# Data loader class to sample from the dataset

from torch.utils.data import DataLoader

train_loader = DataLoader(
        dataset = train_ds,
        batch_size = 2,
        shuffle = True,
        num_workers = 0,
        drop_last = True
        )

test_loader = DataLoader(
        dataset = test_ds,
        batch_size = 2,
        shuffle = False,
        num_workers = 0
        )

# Iterate over the data loader
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
