import os
import re
from math import pi

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rhinoMorph
from gensim.models import Word2Vec
import tensorflow
from tensorflow.compat.v2.keras.models import model_from_json

common_model_url = 'application/model/'
stopword_url = "stopwords-ko_1.txt"
# tokenizer_url = "tokenizer.pickle"
# vz_tokenizer_url = "vz_tokenizer.pickle"

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = "AppleGothic"

plt.switch_backend('Agg')
plt.style.use('ggplot')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_words = 5000
max_len = 100

rn = rhinoMorph.startRhino()
# 이거 mecab으로 바꾸기



# 텍스트 클리닝 - 한글만 남기기
def text_cleaning(doc):
    doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)

    return doc


# 불용어 정의
def define_stopwords(path, encoding):
    SW = set()
    with open(path, encoding=encoding) as f:
        for word in f:
            SW.add(word)

    return SW


# 모델 로딩
def load_model(model_name, model_weight):
    global common_model_url
    json_file = open(common_model_url + model_name, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    # json파일로부터 model 로드하기 
    loaded_model = model_from_json(loaded_model_json)

    # 로드한 model에 weight 로드하기 
    loaded_model.load_weights(common_model_url + model_weight)

    return loaded_model


# 불용어에서 엔터키 제거
SW = define_stopwords(common_model_url + stopword_url, encoding='utf-8')
SW.add('카레')

a = []
for i in SW:
    a.append(i.replace('\n', ''))
SW = set(a)


# tokenizer = joblib.load(common_model_url + tokenizer_url)
# vz_tokenizer = joblib.load(common_model_url + vz_tokenizer_url)


# # 분석!!!
# def sentiment_predict(new_sentence):
#     score = 0
#     model_name = "NLP_model.json"
#     model_weight = "NLP_weight.h5"
#     model = load_model(model_name, model_weight)
#     if new_sentence != '':
#         new_sentence1 = text_cleaning(new_sentence)
#         new_sentence2 = rhinoMorph.onlyMorph_list(rn,new_sentence1, pos = ['NNG', 'NNP','NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi = False)
#         new_sentence3 = [word for word in new_sentence2 if not word in SW] # 불용어 제거
#         encoded = tokenizer.texts_to_sequences([new_sentence3]) # 정수 인코딩
#         if encoded != [[]]:
#             pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
#             score = float(model.predict(pad_new))
#     return score

# # 시각화분석!!!
# def sentiment_predict_VZ(new_sentence, flag):
#     model_name = "NLP_VZ_model.json"
#     model_weight = "NLP_VZ_weight.h5"
#     model = load_model(model_name, model_weight)
#     if new_sentence != '':
#         new_sentence1 = text_cleaning(new_sentence)
#         new_sentence2 = rhinoMorph.onlyMorph_list(rn,new_sentence1, pos = ['NNG', 'NNP','NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi = False)
#         new_sentence3 = [word for word in new_sentence2 if not word in SW] # 불용어 제거
#         encoded = vz_tokenizer.texts_to_sequences(new_sentence3)
#         sum_list = [sum(encoded, [])]
#         if sum_list != [[]]:
#             padded = pad_sequences(sum_list, maxlen = max_len, value = 0, padding = 'pre')
#             #pad_sequence에서 문제
#             tokens = padded
#             #     val = np.array([np.array(tokens)])
#             pred_probs = model.predict(tokens) # 예측

#             categories = ['정서학대', '신체학대', '방임', '성학대']

#             N = len(categories)

#             values = np.round(pred_probs, 3).flatten().tolist()
#             values += values[:1]

#             angles = [n / float(N) * 2 * pi for n in range(N)]
#             angles += angles[:1]

#             plt.polar(angles, values)
#             plt.fill(angles, values)
#             plt.xticks(angles[:-1], categories)

#             plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], color = 'black', size = 8)

#             # plt.show()
#             if flag == 1:
#                 strFile = 'static/images/NLP_VZ_Report_results.png'
#             else:
#                 strFile = 'static/images/NLP_VZ_Diary_results.png'
#             if os.path.isfile(strFile):
#                 os.remove(strFile)   # Opt.: os.system("rm "+strFile)
#                 print("done!")
#             plt.savefig(strFile)
#             plt.close()

# embedding model 인코딩
def encode_sentence_lstm(tokens, emb_size, embedding_model):
    global common_model_url
    embeddings = Word2Vec.load(common_model_url + embedding_model)
    vec = np.zeros((80, 200))
    # maxlen = 80이고 컬럼이 200?
    for i, word in enumerate(tokens):
        if i > 79:
            break
        try:
            vec[i] = embeddings.wv[word].reshape((1, emb_size))
        except KeyError:
            continue
    return vec


# embedding model 분석!!!!
def sentiment_predict_EM(new_sentence):
    score = 0
    model_name = "model-2.json"
    model_weight = "model-2.h5"
    model = load_model(model_name, model_weight)
    if new_sentence != '':
        new_sentence1 = text_cleaning(new_sentence)
        new_sentence2 = rhinoMorph.onlyMorph_list(rn, new_sentence1,
                                                  pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'],
                                                  eomi=False)
        new_sentence3 = [word for word in new_sentence2 if not word in SW]  # 불용어 제거\
        if new_sentence3 != []:
            X = np.array([encode_sentence_lstm(new_sentence3, 200, "Embedding_crawling10000.model")])
            score = float(model.predict(X))  # 예측
    return score


# embedding model visualization
def sentiment_predict_EM_VZ(new_sentence, flag):
    model_name = "Multi_model.json"
    model_weight = "Multi_weight.h5"
    model = load_model(model_name, model_weight)
    if new_sentence != '':
        new_sentence1 = text_cleaning(new_sentence)
        new_sentence2 = rhinoMorph.onlyMorph_list(rn, new_sentence1,
                                                  pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'],
                                                  eomi=False)
        new_sentence3 = [word for word in new_sentence2 if not word in SW]  # 불용어 제거
        if new_sentence3 != []:
            X = np.array([encode_sentence_lstm(new_sentence3, 200, "Multi_Embedding_crawling10000.model")])
            pred_probs = model.predict(X)  # 예측

            categories = ['정서학대', '신체학대', '방임', '성학대']

            N = len(categories)

            values = np.round(pred_probs, 3).flatten().tolist()
            values += values[:1]

            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            plt.polar(angles, values)
            plt.fill(angles, values, alpha=0.4)
            plt.xticks(angles[:-1], categories)

            plt.yticks([0, 0.5, 1.0])
            plt.ylim(0, 1)

            if flag == 1:
                strFile = 'static/images/NLP_VZ_Report_results.png'
            else:
                strFile = 'static/images/NLP_VZ_Diary_results.png'
            if os.path.isfile(strFile):
                os.remove(strFile)  # Opt.: os.system("rm "+strFile)
                print("done!")
            plt.savefig(strFile)
            plt.close()
