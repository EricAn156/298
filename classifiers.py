import json
import numpy as np
import mlp      # our code
import torch
from sklearn import linear_model, model_selection, preprocessing

from torch.utils.data import DataLoader

def gen_data_array(filepath):
    with open(filepath) as f:
        input_json = json.load(f)     # list of length 700
        keys = list(input_json[0].keys())

        # convert the list features into their averages
        for idx, item in enumerate(input_json):
            for k, v in item.items():
                if type(v) == list:
                    input_json[idx][k] = np.mean(v) if len(v) else 0

        # convert the json object into a (700, 16) numpy array
        data = []
        for item in input_json:
            row = np.array([v for _, v in item.items()])
            data.append(row)
        data = np.array(data)
        return keys, data
    

def gen_full_dataset():
    automated_filepath = './automated/automatedAccountData.json'
    nonautomated_filepath = './automated/nonautomatedAccountData.json'

    keys, automated_data = gen_data_array(automated_filepath)
    keys, nonautomated_data = gen_data_array(nonautomated_filepath)
    # print(keys, automated_data.shape, automated_data.dtype, nonautomated_data.shape, nonautomated_data.dtype)

    # the 'automatedBehaviour' key is the ground truth value
    data = np.vstack((automated_data, nonautomated_data))
    np.random.shuffle(data)

    gt_index = keys.index('automatedBehaviour')
    x = data[:, 0:gt_index]
    y = data[:, gt_index]
    return x, y


def train_linear_classifier(x_train, y_train):
    model = linear_model.LinearRegression().fit(x_train, y_train)
    return model

# def train_MLP(x_train, y_train):
#     model = 


def test_model(name, model, threshold, x_test, y_test):
    y_test_pred = model.predict(x_test)
    mask = y_test_pred > threshold
    y_test_pred = np.zeros_like(y_test_pred)
    y_test_pred[mask] = 1

    print(name, np.count_nonzero(y_test_pred == y_test), 'correct, out of', len(y_test), ', accuracy:', np.count_nonzero(y_test_pred == y_test)/len(y_test))
    return model


if __name__ == '__main__':
    np.random.seed(666)
    x, y = gen_full_dataset()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    # model outputs are a probability between 0 and 1, threshold for bot prediction: 0.5
    # threshold = 0.5
    # linear = train_linear_classifier(x_train, y_train)
    # test_model('linear classifier', linear, threshold, x_test, y_test)

    # normalize data for neural net
    x_train = mlp.normalize_data(x_train)
    x_test = mlp.normalize_data(x_test)

    batch_size = 64
    train_dataset = mlp.BotDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = mlp.BotDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # for training the model and saving parameters
    model_path = 'mlp_model.pth'
    mlp_model, loss_graph = mlp.train_MLP(train_loader, test_loader, len(test_dataset), input_features=x_train.shape[1])
    loss_path = 'loss_graph.txt'
    with open(loss_path, 'w') as f:
        for item in loss_graph:
            f.write(f'{item[0]}, {item[1]}, {item[2]}\n')

    torch.save(mlp_model.state_dict(), model_path)

    # # for uploading previously saved parameters
    # # mlp_model = mlp.MLP(input_features=x_train.shape[1])
    # # mlp_model.load_state_dict(torch.load(save_path))
    # # mlp.test_MLP('MLP', test_loader, mlp_model, threshold, len(test_dataset))

