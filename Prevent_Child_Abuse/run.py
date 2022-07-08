from flask import request, render_template, redirect, Flask

import application.ml.model as MM  # 정형데이터처리 연동
import application.ml.NLP as nlp  # 자연어처리 연동
import application.ml.Database as db  # 데이터베이스 연동

from werkzeug.utils import secure_filename

import random

app = Flask(__name__)


# app = Blueprint('main', __name__, url_prefix='/')

# 메인페이지 라우팅
@app.route("/")
@app.route('/main')
def index():
    return render_template('/main/index.html')


# 소개페이지 라우팅
@app.route('/about')
def about():
    return render_template('/About Us/about.html')


# Folium 시각화페이지 라우팅
@app.route('/results')
def results():
    return render_template('/Maps/map.html')


# 새로운 데이터 입력 페이지 라우팅
@app.route('/putData')
def putData():
    return render_template('/Test/putData.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        input_data = []
        temp_input_data = []
        data = [0 for i in range(77)]
        # 아동_성별
        input_data.append(request.form['sex'])
        # 아동_생년월일
        temp_birthday = request.form['birthday']
        # 생년월일을 나이대로 구분해주기
        if temp_birthday == '':
            birthday = None
        else:
            birthday = int(temp_birthday)
            if (2019 - (birthday / 10000)) >= 0.0 and (2019 - (birthday / 10000)) < 1.0:
                input_data.append('아동_생년월일_0~1세미만')
            elif (2019 - (birthday / 10000)) >= 1.0 and (2019 - (birthday / 10000)) <= 3.0:
                input_data.append('아동_생년월일_1~3세')
            elif (2019 - (birthday / 10000)) >= 4.0 and (2019 - (birthday / 10000)) <= 6.0:
                input_data.append('아동_생년월일_4~6세')
            elif (2019 - (birthday / 10000)) >= 7.0 and (2019 - (birthday / 10000)) <= 9.0:
                input_data.append('아동_생년월일_7~9세')
            elif (2019 - (birthday / 10000)) >= 10.0 and (2019 - (birthday / 10000)) <= 12.0:
                input_data.append('아동_생년월일_10~12세')
            elif (2019 - (birthday / 10000)) >= 13.0 and (2019 - (birthday / 10000)) <= 15.0:
                input_data.append('아동_생년월일_13~15세')
            elif (2019 - (birthday / 10000)) >= 16.0 and (2019 - (birthday / 10000)) <= 17.0:
                input_data.append('아동_생년월일_16~17세')
            else:
                input_data.append('아동_생년월일_18~20세')
        # 아동_내국인여부
        input_data.append(request.form['domestic'])
        # 아동_최종학력
        input_data.append(request.form['edu'])
        # 아동_직업유형
        input_data.append(request.form['job'])
        # 아동_거주상태
        input_data.append(request.form['residence'])
        # 아동_친권자유형
        input_data.append(request.form['parents'])
        # 아동_가족유형
        input_data.append(request.form['family'])
        # 아동_다문화가족
        input_data.append(request.form['multi_culture'])
        # 아동_가구소득구분코드
        input_data.append(request.form['income'])
        # 아동_기초생활수급유
        input_data.append(request.form['supply'])
        # 신고_접수경로구분코
        input_data.append(request.form['route'])
        # 신고_신고자유형구분
        input_data.append(request.form['reporter'])
        # 신고_집단시설내사건
        input_data.append(request.form['group'])
        # 신고_재신고여부_1
        input_data.append(request.form['re'])
        # 신고_접수유형
        input_data.append(request.form['report_type_1'])
        # 신고_피해아동상태구
        input_data.append(request.form['child'])
        # 신고_행위자아동관계
        input_data.append(request.form['relationship'])
        # 신고_아동동거여부
        input_data.append(request.form['together'])
        # 신대_접수유형
        input_data.append(request.form['report_type'])
        # 아동특성 79가지
        temp_input_data.append(request.form.getlist('characteristic'))

        for i in range(len(temp_input_data[0])):
            input_data.append(temp_input_data[0][i])

        print(input_data)

        model_results = MM.report_expectation(input_data)

        # 신고 접수내용 자연어처리
        Report_text = request.form['text']
        text_NLP_results = nlp.sentiment_predict_EM(Report_text)

        # 학대 유형 시각화, flag 1은 신고접수 2는 다이어리
        flag = 1
        nlp.sentiment_predict_EM_VZ(Report_text, flag)

        # 시각화 자료 불러오기 -> 이미지 캐싱문제 해결위해 랜덤번호 붙여줌
        source_url = 'static/images/NLP_VZ_Report_results.png'
        random_num = int(random.random() * 1000000)
        url = source_url + '?ver=' + str(random_num)

        # # 결과 리턴
        return render_template('/Test/putData.html', model_results=model_results, text_NLP_results=text_NLP_results,
                               input_data=input_data, Report_text=Report_text, name='학대 유형', url=url)


# ID기반 조회
@app.route('/searchID', methods=['GET', 'POST'])
def searchID():
    db_ID_data = []
    len_ID_data = 0
    predict_results = []
    if request.method == 'POST':
        IDnumber = request.form['ID']
        db_ID_data = db.read_IDdata(IDnumber)
        db_data_dummy = MM.make_db_data_dummy(db_ID_data)
        reabuse_predict = []
        for i in range(len(db_data_dummy)):
            reabuse_predict.append(MM.model_avg(db_data_dummy.iloc[i, :]))
        predict_results = db_ID_data[['피해아동대상자', '아동_성별', '아동_생년월일', '신고_접수일시', '신대_통계거점', 'NEW_CALL_COUNT']]
        predict_results['재학대 발생확률'] = reabuse_predict
        len_ID_data = len(predict_results.index)
    return render_template('/Test/ID.html', db_ID_data=predict_results, len_ID_data=len_ID_data)


# 재학대 예측
@app.route('/inquire', methods=['GET', 'POST'])
def inquire():
    db_data = []
    predict_results = []
    df_length = 0
    # 조회버튼 클릭시
    if request.method == 'POST':
        # 현재 위치(IP기반), 시간 기반으로 데이터베이스 조회
        db_data = db.read_database()
        db_data_dummy = MM.make_db_data_dummy(db_data)
        reabuse_predict = []
        for i in range(len(db_data_dummy)):
            reabuse_predict.append(MM.model_avg(db_data_dummy.iloc[i, :]))
        predict_results = db_data[['피해아동대상자', '아동_성별', '아동_생년월일', '신고_접수일시', '신대_통계거점', 'NEW_CALL_COUNT']]
        predict_results['재학대 발생확률'] = reabuse_predict
        df_length = len(predict_results.index)

        # 읽은 데이터 모델 돌려야함
        # 모델 언제나와융
    return render_template('/Test/inquire.html', inquire_results=predict_results, df_length=df_length)


# 다이어리 통한 분석
@app.route('/diary', methods=['GET', 'POST'])
def diary():
    # 다이어리 내용 입력시
    if request.method == 'POST':
        diary_text = request.form['diary_textarea']
        # 아동학대 확률 예측
        diary_NLP_results = nlp.sentiment_predict_EM(diary_text)

        # 학대 유형 시각화, flag 1은 신고접수 2는 다이어리
        flag = 2
        # nlp.sentiment_predict_VZ(diary_text, flag)
        nlp.sentiment_predict_EM_VZ(diary_text, flag)

        # 시각화 자료 불러오기 -> 이미지 캐싱문제 해결위해 랜덤번호 붙여줌
        source_url = 'static/images/NLP_VZ_Diary_results.png'
        random_num = int(random.random() * 1000000)
        url = source_url + '?ver=' + str(random_num)
        return render_template('/Diary/diary.html', diary_text=diary_text, diary_NLP_results=diary_NLP_results,
                               name='학대 유형', url=url)

    else:
        diary_text = None
        diary_NLP_results = 0.0

    return render_template('/Diary/diary.html', diary_text=diary_text, diary_NLP_results=diary_NLP_results)


# 그림???
@app.route('/sketch', methods=['GET', 'POST'])
def sketch():
    if request.method == 'POST':
        # 업로드 파일 처리 분기
        file = request.files['image']
        sfname = 'static/images/' + str(secure_filename(file.filename))
        if not file:
            sketch_text = 'No Files'
            return render_template('/Sketch/sketch.html', label=sketch_text)
        else:
            sketch_text = '제출완료'
            return render_template('/Sketch/sketch.html', label=sketch_text, imgpath=sfname)

    return render_template('/Sketch/sketch.html')


if __name__ == "__main__":
    app.debug = True

    app.run(host="0.0.0.0", port=5050)
