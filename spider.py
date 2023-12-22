import requests
import pandas as pd
import time
import json
import csv
import copy

def spider():
    #下面是爬取某学期（2023-2024-1）的某学院（安泰经济与管理学院）所开设的课程
    url = 'https://plus.sjtu.edu.cn/course-plus-data/lessonData_2023-2024_2.json'
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}

    response = requests.get(url)


    if response.status_code == 200:

        json_data = response.json()

        found_records = []


        for record in json_data:
            if "安泰经济与管理学院" in str(record):
                found_records.append(record)

        
        df = pd.DataFrame(found_records)
        df = df[["kcmc", "kch", "zjs", "xf", "cdmc", "sksj", "nj", "kcxzmc"]]
        df = df.rename(columns={"kcmc":"课程名称", "kch":"课程号","zjs":"任课教师", "cdmc":"上课地点", "sksj":"上课时间", 
                                "nj":"年级", "kcxzmc":"课程性质", "xf":"学分"})

        print("CSV文件已保存")
    else:
        print("请求失败")


    #下面是将选课社区的评分增加在已经生成的csv文件中并增加为“评分”列
    url2 = 'https://course.sjtu.plus/api/course/?&department=89&page=1&size=10000'
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    cookies = {
        "__yjs_duid": "1_fd743bb0dcff89f4d93fa8ece1e0b0091676547479153",
        "csrftoken": "4ft2z3Pmca27c7HQJUph0gkauOqc3zv0",
        "sessionid": "0z8d9pfrya5kv2qmvbcptm9fwd30p338",
        "Hm_lvt_bffe2d130d940fce5a0876ee2dc36b92": "1700566119,1702270409",
        "Hm_lpvt_bffe2d130d940fce5a0876ee2dc36b92": "1702303152"
    }

    url2 = 'https://course.sjtu.plus/api/course/?&department=89&page=1&size=10000'

    response2 = requests.get(url2, headers=headers, cookies=cookies)
    html = response2.text


    parsed_html = json.loads(html)
    course_data = parsed_html.get("results", [])
    courses = []

    for course in course_data:
        course_dict = {
            'teacher': course.get('teacher'),
            'rating': course.get('rating'),
            'code': course.get('code'),
            'name': course.get('name'),
            'credit': course.get('credit')
        }
        
        course['rating'].pop('count')
        course['rating'] = course['rating']['avg']
        
        courses.append(course_dict)
    print(courses)

    df["选课社区评分"] = 0.0

    teacher_list = df['任课教师'].tolist()
    name_list = df['课程名称'].tolist()

    for i in range(len(df)):
        teacher_tg = teacher_list[i]
        name_tg = name_list[i]
        
        for course in courses:
            if teacher_tg == course['teacher'] and name_tg == course['name']:
                df.loc[i, "选课社区评分"] = course['rating']['avg']

    # 筛选年级
    df = df[df["年级"] == '2022']

    # TO DO
    df["概率"] = 1

    df.to_csv('found_records.csv', index=False)

def get_data():
    # df1 = pd.read_csv('found_records_manual.csv', encoding='utf-8-sig')
    df1= pd.read_csv('found_records_manual.csv', encoding='utf-8')
    df1 = df1[df1["年级"] == '2022']
    # df1["概率"] = 1
    df2 = copy.deepcopy(df1)
    df2 = df2.drop_duplicates("课程名称", keep='first').reset_index(drop=True)
    return df1, df2
