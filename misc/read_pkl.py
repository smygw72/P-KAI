import pickle


if __name__ == '__main__':
    with open('./annotation/annotations_unidist_train.pkl', 'rb') as file:
        unidist_train = pickle.load(file)

    with open('./annotation/annotations_consecutive_train.pkl', 'rb') as file:
        consecutive_train = pickle.load(file)

    print(unidist_train)
    print(consecutive_train)
