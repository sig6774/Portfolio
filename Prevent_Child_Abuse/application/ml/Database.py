import os

import cx_Oracle
import geocoder
import pandas as pd
from geopy.geocoders import Nominatim

os.environ["NLS_LANG"] = ".AL32UTF8"
location_list = ['경기', '강원', '경상남도', '경상북도', '충청남도', '충청북도', '전라남도', '전라북도', '제주특별자치도', '대전', '광주', '부산', '울산', '대구',
                 '인천']


# DB내용 출력, 테스트용
def print_db():
    cursor.execute(sql.encode('utf-8'))
    for row in cursor:
        print(row)


# 위치기반, 현재 날짜로부터 3~6개월전 데이터 조회
def read_database():
    # DB 연결
    conn = cx_Oracle.connect("Jason/ssis1234@localhost:49161/xe")
    # cursor = conn.cursor()

    # 위치정보
    myloc = geocoder.ip('me')
    locator = Nominatim(user_agent="myGeocoder")
    coordinates = "%s, %s" % (myloc.latlng[0], myloc.latlng[1])
    print(coordinates)
    location = locator.reverse(coordinates)
    my_location = None

    try:
        my_location = location.raw['address']['city']
    except(KeyError):
        pass
    try:
        my_location = location.raw['address']['province']
    except(KeyError):
        pass

    for i in range(len(location_list)):
        if my_location != location_list[i]:
            if (myloc.latlng[0] >= 37.413294 and myloc.latlng[0] <= 37.715133) and (
                    myloc.latlng[1] >= 126.734086 and myloc.latlng[1] <= 127.269311):
                my_location = '서울'
            elif (myloc.latlng[0] >= 36.418608 and myloc.latlng[0] <= 36.733585) and (
                    myloc.latlng[1] >= 127.126739 and myloc.latlng[1] <= 127.409310):
                my_location = '세종'
    if my_location == '경기도':
        my_location = '경기'
    elif my_location == '경상남도':
        my_location = '경남'
    elif my_location == '경상북도':
        my_location = '경북'
    elif my_location == '충청남도':
        my_location = '충남'
    elif my_location == '충청북도':
        my_location = '충북'
    elif my_location == '전라남도':
        my_location = '전남'
    elif my_location == '전라북도':
        my_location = '전북'
    elif my_location == '제주특별자치도':
        my_location = '제주'

    sql_read = """select * from child_model where (신대_통계거점 = '%s' and ROWNUM < 16) order by 신고_접수일시 ASC""" % my_location

    db_data = pd.read_sql(sql_read, conn)
    conn.close()

    return db_data


def read_IDdata(temp_ID):
    # DB 연결
    conn = cx_Oracle.connect("Jason/ssis1234@localhost:49161/xe")

    sql_read = """select * from child_model where (피해아동대상자 = '%s')""" % temp_ID

    db_ID_data = pd.read_sql(sql_read, conn)
    conn.close()

    return db_ID_data
