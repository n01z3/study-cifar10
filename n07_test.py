import numpy as np
from n06_train import *
from n02_eda import *
from n04_dataset import *
import torch


def main():
    os.makedirs('subm', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)
    state = torch.load('weights/model_best.pth.tar')
    net.load_state_dict(state)

    _, _, testloader = get_loaders(256)

    y_preds = []
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader)):
            inputs = batch['image'].to(device)
            # labels = batch['y'].to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            sample_predict = predicted.cpu().numpy()
            y_preds.append(sample_predict)

    y_preds = np.concatenate(y_preds, axis=0)

    test_df = pd.read_csv(os.path.join(DATA['path'], DATA['sample']))
    test_df['label'] = [idx2klass.get(el) for el in y_preds]
    test_df.to_csv('subm/subm1.csv', index=False)


if __name__ == '__main__':
    main()
