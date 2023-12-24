from spider import *
from utils import *
from runner import get_course

def test(course_list):
    """
    本函数旨在比较不同算法的性能，在filtering层面，比较了naive，forward checking和AC3算法的性能
    在ordering层面，比较了random和MRV(Minimum Remaining Values)两种方法
    """
    n = len(course_list)
    print('n = ', n)
    # 采用默认的用户偏好
    courses, teachers = get_course()
    userPreference = UserPreference()
    for course in courses:
        # 默认用户无偏好
        if course.name in course_list:
            userPreference.add_courses_preference(course, 1)
    selected_courses = [x[0] for x in userPreference.courses_preference]
    solver_naive = optimal_naive_Solver(selected_courses, userPreference, teachers)
    solver_forward = optimal_forward_checking_Solver(selected_courses, userPreference, teachers)
    solver_AC3 = optimal_AC3_Solver(selected_courses, userPreference, teachers)
    for i in range(3):
        schedule = Schedule()
        if i == 0:
            solver = solver_naive
            print("-----------------naive-----------------")
        elif i == 1:
            solver = solver_forward
            print("-----------------forward checking-----------------")
        else:
            solver = solver_AC3
            print("-----------------AC3-----------------")

        optimal_schedule = solver.solve(schedule, 2)
        print("explored: ", solver.explored)
    

if __name__ == '__main__':
    # test()
    course_list_1 = ['金融学原理', '计量经济学', '货币金融学']
    course_list_2 = ['金融学原理', '计量经济学', '货币金融学', '博弈论', '数据采集与可视化']
    course_list_3 = ['金融学原理', '计量经济学', '货币金融学', '博弈论', '数据采集与可视化','机器学习', '回归分析', '经济优化方法', '运营管理', '定价策略']
    test(course_list_1)
    test(course_list_2)
    test(course_list_3)


