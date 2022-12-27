import pickle


MBTI_type = {
    'INFJ': 0,
    0: 'INFJ',
    'INFP': 1,
    1: 'INFP',
    'INTJ': 2,
    2: 'INTJ',
    'INTP': 3,
    3: 'INTP',
    'ISFJ': 4,
    4: 'ISFJ',
    'ISFP': 5,
    5: 'ISFP',
    'ISTJ': 6,
    6: 'ISTJ',
    'ISTP': 7,
    7: 'ISTP',
    'ENFJ': 8,
    8: 'ENFJ',
    'ENFP': 9,
    9: 'ENFP',
    'ENTJ': 10,
    10: 'ENTJ',
    'ENTP': 11,
    11: 'ENTP',
    'ESFJ': 12,
    12: 'ESFJ',
    'ESFP': 13,
    13: 'ESFP',
    'ESTJ': 14,
    14: 'ESTJ',
    'ESTP': 15,
    15: 'ESTP'
}

class LogisticRegression:
    def __init__(self):
        with open('static/lr/vectorizer.pickle', 'rb') as file:
            self.vectorizer = pickle.load(file)

        # with open('colab/lr_scaler.pickle', 'rb') as file:
        #     scaler = pickle.load(file)
        # X_rescaled = scaler.fit_transform(X)

        # with open('colab/lr_pca.pickle', 'rb') as file:
        #     pca = pickle.load(file)
        # pca.fit(X_rescaled)
        # self.X_pca = pca.transform(X_rescaled)

        with open('static/lr/model.pickle', 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, input_post):
        X = self.vectorizer.transform([input_post]).toarray()
        prediction = self.model.predict(X)[0]
        return MBTI_type[prediction]


lr_model = LogisticRegression()