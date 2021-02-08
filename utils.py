import pandas as pd


def category_to_index(csv_path='./data/Question_Classification_Dataset.csv'):
    df = pd.read_csv(csv_path)
    label = df['Category0']
    category = set(label.values)
    category = sorted(category)
    category = {word: index for index, word in enumerate(category)}
    return category


def get_data(csv_path='./data/Question_Classification_Dataset.csv'):
    df = pd.read_csv(csv_path)

    questions = df['Questions']
    questions = list(questions)

    category = category_to_index(csv_path=csv_path)
    labels = [category[i] for i in df['Category0'].values]
    labels = list(labels)

    return questions, labels


if __name__ == '__main__':
    print(category_to_index())
    pass
