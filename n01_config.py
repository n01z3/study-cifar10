import yaml


def get_paths(path="config/path.yml"):
    with open(path, 'r') as stream:
        data_config = yaml.load(stream)
    return data_config


def get_params(path="config/train.yml"):
    with open(path, 'r') as stream:
        data_config = yaml.load(stream)
    return data_config


if __name__ == '__main__':
    # path = get_paths()
    # print(path['dataset'])
    #
    # data = path['dataset']
    #
    # dataset_dir = data['path']
    # print(dataset_dir)
    params = get_params()
    print(params)