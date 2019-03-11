from n02_eda import *


def main():
    df = pd.read_csv(os.path.join(DATA['path'], DATA['lables_csv']))
    print(df.head())

    os.makedirs('tables', exist_ok=True)

    df.sort_values('label', inplace=True)
    df['fold_id'] = ([0, 1, 2, 3, 4] * df.shape[0])[:df.shape[0]]

    print(df.head())
    df.to_csv(df.to_csv('tables/folds_n01.csv', index=False))


def prepare_test():
    df = pd.read_csv(os.path.join(DATA['path'], DATA['sample']))
    print(df.head())


if __name__ == '__main__':
    # main()
    prepare_test()
