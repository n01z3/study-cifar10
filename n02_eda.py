import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from n01_config import get_paths

DATA = get_paths()['dataset']
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    df = pd.read_csv(os.path.join(DATA['path'], DATA['lables_csv']))
    print(df.head())

    labels = sorted(set(df['label'].tolist()))
    print(labels)

    for label in LABELS:
        tdf = df[df['label'] == label]
        print(f'{label}: {tdf.shape[0]}')

    label = LABELS[0]
    tdf = df[df['label'] == label]
    tdf.reset_index(inplace=True, drop=True)

    plt.figure()
    for i in range(9):
        plt.subplot(3,3,1+i)
        filename = f'{tdf.loc[i, "id"]}.png'
        img = cv2.imread(os.path.join(DATA['path'], DATA['train_dir'], filename))[:,:,::-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    main()
