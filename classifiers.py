import json
import numpy as np
from sklearn import datasets, linear_model, model_selection, metrics

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

def train_linear_classifier(x_train, x_test, y_train, y_test):
    # model outputs are a probability between 0 and 1, threshold for bot prediction: 0.5
    threshold = 0.5
    model = linear_model.LinearRegression().fit(x_train, y_train)

    y_test_pred = model.predict(x_test)
    mask = y_test_pred > threshold
    y_test_pred = np.zeros_like(y_test_pred)
    y_test_pred[mask] = 1

    print('linear classifier:', np.count_nonzero(y_test_pred == y_test), 'correct, out of', len(y_test), ', accuracy:', np.count_nonzero(y_test_pred == y_test)/len(y_test))
    return model


if __name__ == '__main__':
    np.random.seed(666)
    x, y = gen_full_dataset()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    linear = train_linear_classifier(x_train, x_test, y_train, y_test)

    

