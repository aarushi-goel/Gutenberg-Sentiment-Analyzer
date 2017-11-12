import pandas as pd
from NaiveBayesAnalyzer import NaiveBayesAnalyzer


def create_dict(path):
    df = pd.read_csv(path, usecols=(1, 3))
    df2 = df.tail(len(df) - 8000)
    df = df.head(32000)
    sentiments = set([(v['sentiment']) for k, v in df.iterrows()])

    train_dic = dict()
    for i in sentiments:
        train_dic[i] = []

    for k, v in df.iterrows():
        train_dic[v['sentiment']].append(v['content'])

    return train_dic, df2


def main():
    dic, testdata = create_dict('raw_data/text_emotion.csv')
    analyzer = NaiveBayesAnalyzer(dic)
    analyzer.train()

    count = 0
    for k,v in testdata.iterrows():
        predicted = analyzer.analyze(v['content'])
        if predicted == v['sentiment']:
            count+=1

    print (count/len(testdata))


if __name__ == '__main__':
    main()
