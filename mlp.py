import torch
# from torchsummary import summary
from torch import nn
from torch.utils.data import Dataset, DataLoader

class BotDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]        
        sample = {'features': x, 'class': y}
        return sample
    
class MLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 2, dtype=torch.float64),
            # nn.ReLU(),
            # nn.Linear(32, 1, dtype=torch.float64),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
def train_MLP(train_loader, test_loader, test_length, input_features):
    model = MLP(input_features)
    # print(summary(model, input_size=()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 7500
    loss_graph = []
    max_accuracy = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data['features'], data['class']
            labels = labels.long()
            # print('shapes', inputs.shape, labels.shape, labels)
            # inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.dtype, outputs.shape, labels.dtype, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # for layer in model.parameters():
            #     print(layer.grad)

            running_loss += loss.item()
        if epoch % 100 == 0:
            print(f'-----Epoch {epoch} loss: {running_loss}')
            test_accuracy = test_MLP('MLP', test_loader, model, test_length)
            max_accuracy = max(max_accuracy, test_accuracy)
            print('max accuracy:', max_accuracy)
            loss_graph.append((epoch, running_loss, test_accuracy))
        running_loss = 0.0

    return model, loss_graph


def test_MLP(name, test_loader, model, total_samples):
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['features'], data['class']
            outputs = model(inputs)
            
            # probabilities = torch.softmax(outputs, dim=0)
            # preds = torch.zeros_like(probabilities)
            # preds[probabilities > threshold] = 1
            preds = torch.argmax(outputs, dim=1)
            # print('shape', preds.shape, labels.shape)
            # print('correct', torch.sum(preds == labels))
            correct += torch.sum(preds == labels).item()
        
        accuracy = correct / total_samples
        print(name, correct, 'correct out of', total_samples, ', accuracy:', accuracy)
    return accuracy


# if __name__ == '__main__':
