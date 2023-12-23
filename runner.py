import sys
import interface
import out
import pandas as pd
import spider
import itertools
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QMessageBox,QWidget,QDialog
from PyQt5 import QtCore
from utils import *
import os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--is_update', type=bool, default=False, help='whether to update the model')
# 解析命令行参数
args = parser.parse_args()
is_update = args.is_update


wd = os.path.dirname(__file__)
os.chdir(wd)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QApplication(sys.argv)


mainWindow = QMainWindow()

ui = interface.Ui_MainWindow()
ui.setupUi(mainWindow)


MainWindow = QMainWindow()
out_interface = out.Ui_MainWindow()
out_interface.setupUi(MainWindow)

#MainWindow.show()

def load_course():
    df1, df2 = spider.get_data()

    # 读取课程列表，并添加到课程偏好的选项中
    courses_name = list(set(df1["课程名称"]))
    key = 0
    for course in courses_name:
        ui.comboBox_7.addItem("")
        ui.comboBox_7.setItemText(key,course)
        key += 1
        
    return None

def get_course():
    """读取课程信息"""
    df1, df2 = spider.get_data()
    courses = []
    all_teachers= dict()

    for row in df2.itertuples():
        courses.append(Course(row[2], row[1], int(row[4]), []))
    for row in df1.itertuples():
        for course in courses:
            if course.id == row[2]:
                time_slot = row[6].split('|')[0]
                teacher = row[3]
                course.sections.append(Section(course, time_slot, float(row[9]), float(row[11]), teacher, all_teachers))
                break
    return courses, all_teachers

def get_rewards():
    labels = []    # load data

    while True:
        answer, ok = QInputDialog.getText(None, '输入', '您是否认为第一个课表优于第二个课表？\n输入y或n')
        if ok:
            if answer not in ['y', 'n']:
                QMessageBox.information(None, '警告', '输入有误，请重新输入')
                continue
            else:
                if answer == 'y':
                    labels = [1, 2]
                else:
                    labels = [2, 1]
                break
    return labels


#打开某一个偏好
def showPreferredTime():
    ui.widget_2.show()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_7.hide()

def showDislikedTime():
    ui.widget_2.hide()
    ui.widget_3.show()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_12.hide()

def showPreferredTeacher():
    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.show()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_14.hide()

def showSelectiveCourses():
    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.show()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_16.hide()

def showPreferredProb():
    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.show()
    ui.widget_7.hide()
    ui.label_19.hide()

def showCoursePreference():
    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.show()
    ui.label_23.hide()


def confirmAll():
    
    courses, teachers = get_course()

    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_17.show()

    # 处理偏好
    userpreference = UserPreference()
    '''
    for time in preferredTime:
        userpreference.add_preferred_time(time[0], time[1], time[2])
    for time in dislikedTime:
        userpreference.add_not_preferred_time(time[0], time[1], time[2])
    for teacher in preferredTeacher:
        userpreference.add_preferred_teacher(teacher)
    
    
    for pref in coursePreference:
        for course in courses:
            if course.name == pref[0] and pref[1] > 0:
                userpreference.add_courses_preference(course, pref[1])
    '''
    for ru in ui.content:
        if ru[0] == 'like_time':
            userpreference.add_preferred_time(int(ru[1][0]),int(ru[1][1]),int(ru[1][2]))
        elif ru[0] == 'dislike_time':
            userpreference.add_not_preferred_time(int(ru[1][0]),int(ru[1][1]),int(ru[1][2]))
        elif ru[0] == 'like_teacher':
            # print('adding a teacher!')
            userpreference.add_preferred_teacher(ru[1])
        elif ru[0] == 'like_course':
            for course in courses:
                if course.name == ru[1][0] and ru[1][1] > 0:
                    userpreference.add_courses_preference(course, int(ru[1][1]))
    userpreference.set_num_selective(numSelective)
    userpreference.set_prob_preference(preferredProb)


    # 更新出用户要选的课
    selected_courses = [x[0] for x in userpreference.courses_preference]

    # 拷贝courses和teachers
    # courses_copy = copy.deepcopy(selected_courses)
    # new_teachers = dict()
    # for course in courses_copy:
    #     for section in course.sections:
    #         if section.teacher not in new_teachers:
    #             new_teachers[section.teacher] = [section]
    #         else:
    #             new_teachers[section.teacher].append(section)

    # solver = optimal_naive_Solver(courses_copy, userpreference, teachers)
    # solver = optimal_forward_checking_Solver(courses_copy, userpreference, teachers)
    solver = optimal_AC3_Solver(selected_courses, userpreference, teachers)
    schedule = Schedule()
    optimal_schedule = solver.solve(schedule, 2, True)
    if len(optimal_schedule)==0:
        QMessageBox.information(None, '提示', '没有可行的课表，请重新输入!')
        return
    #QMessageBox.information(None, '提示', '已生成课表，请在out文件夹中查看！')
    schedule_rating = optimal_schedule[0].get_schedule_rating()
    print(f'--------------- rating: {schedule_rating} --and--explored: {solver.explored} ---------------')
    #optimal_schedule[0].draw()

    for i in os.listdir('out'):
        os.remove('out\\'+i)
    for i in range(len(optimal_schedule)):
        optimal_schedule[i].draw(i+1)

    #### 是否要更新模型，如果需要，请在运行程序时加上参数 --is_update=True
    if is_update:
        rewards = get_rewards()
        solver.update_model(rewards)
        if rewards == [1, 2]:
            QMessageBox.information(None, '提示', '谢谢您的反馈！')
        else:
            QMessageBox.information(None, '提示', '谢谢您的反馈，模型已更新')

    
    MainWindow = QDialog()
    out_interface = out.Ui_MainWindow()
    out_interface.setupUi(MainWindow)
    MainWindow.exec()
    #sys.exit(app.exec_())
    #### 是否要更新模型，如果需要，请在运行程序时加上参数 --is_update=True
    if is_update:
        rewards = get_rewards()
        solver.update_model(rewards)
        if rewards == [1, 2]:
            QMessageBox.information(None, '提示', '谢谢您的反馈！')
        else:
            QMessageBox.information(None, '提示', '谢谢您的反馈，模型已更新')


if __name__ == '__main__':
    # 载入课程
    load_course()   
    
    #初始化所有widget都隐藏
    ui.widget_2.hide()
    ui.widget_3.hide()
    ui.widget_4.hide()
    ui.widget_5.hide()
    ui.widget_6.hide()
    ui.widget_7.hide()
    ui.label_17.hide()
    # 读取用户给的数据
    preferredTime=list()
    dislikedTime=list()
    preferredTeacher=list()
    numSelective=0
    preferredProb=0
    coursePreference=list()
    def readPreferredTime():
        preferredTime.append((int(ui.comboBox.currentText()),int(ui.comboBox_2.currentText()),int(ui.comboBox_3.currentText())))
    def readDislikedTime():
        dislikedTime.append((int(ui.comboBox_6.currentText()),int(ui.comboBox_4.currentText()),int(ui.comboBox_5.currentText())))
    def readPreferredTeacher():
        preferredTeacher.append(ui.textEdit.toPlainText())
    def readNumSelective():
        global numSelective
        numSelective=int(ui.spinBox.value())
    def readpreferredProb():
        global preferredProb
        preferredProb=float(ui.textEdit_2.toPlainText())
    def readCoursePreference():
        coursePreference.append((ui.comboBox_7.currentText(), int(ui.spinBox_2.value())))

    # 切换界面
    ui.pushButton.clicked.connect(showPreferredTime)
    ui.pushButton_2.clicked.connect(showDislikedTime)
    ui.pushButton_3.clicked.connect(showPreferredTeacher)
    # ui.pushButton_4.clicked.connect(showSelectiveCourses)
    ui.pushButton_13.clicked.connect(showPreferredProb)
    ui.pushButton_14.clicked.connect(showCoursePreference)
    ui.pushButton_12.clicked.connect(confirmAll)

    # 按键控制读取
    ui.pushButton_6.clicked.connect(readPreferredTime)
    ui.pushButton_8.clicked.connect(readDislikedTime)
    ui.pushButton_10.clicked.connect(readPreferredTeacher)
    # ui.pushButton_11.clicked.connect(readNumSelective)
    ui.pushButton_15.clicked.connect(readpreferredProb)
    ui.pushButton_17.clicked.connect(readCoursePreference)

    mainWindow.show()

    sys.exit(app.exec_())