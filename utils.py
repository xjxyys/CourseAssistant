import numpy as np
from constraint import *
import copy
import random
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import os

#####################################################################
#以下是本项目使用到的数据结构
#####################################################################

class Course:
    def __init__(self, id:str, name: str, credit:int, sections:list) -> None:
        self.id = id  # 课程ID e.g.BUSS3620
        self.name = name  # 课程名称 e.g.人工智能导论
        self.credit = credit  # 学分 e.g.3
        self.sections = sections  # 教学班列表

    def __str__(self) -> str:
        return f"{self.name} - Sections: {len(self.sections)}"


class Section:
    def __init__(self, course: Course, time_slot: str, initial_rating: float, prob: float, teacher:str, all_teachers:dict) -> None:
        self.course = course # 属于哪个课程
        self.time_slot = time_slot  # 时间安排 星期二第11-13节{1-16周}
        self.teacher = teacher # 课程老师
        self.initial_rating = initial_rating  # 选课社区评分 e.g. 5.0
        self.prob = prob # 选上课的概率
        self.rating = initial_rating  # 评分，初始值为选课社区评分

        self.features = [self.initial_rating, 0, 0, 0] # 特征向量(构造的评分，是否喜欢老师，是否喜欢这个时间，课程偏好)

        all_teachers[self.teacher] = all_teachers.get(self.teacher, []) + [self] # 记录该section的老师
        # 处理时间安排，整合成slots
        d = dict(zip(['一', '二', '三', '四', '五'], list(range(1, 6))))
        self.slots = []

        # 星期二第1-2节{1-16周};星期四第3-4节{2-16周(双)}
        for time in time_slot.split(';'):
            day = d[time[2]]

            start, end = map(int, time.split('第')[1].split('节')[0].split('-'))
            times = list(range(start, end + 1))

            week_str = time.split('第')[1].split('节')[1].split('{')[1].split('周')[0]
            if '(' in week_str:
                # 处理特殊周次，如“2-16(双)”
                base_range, pattern = week_str.split('(')
                start, end = map(int, base_range.split('-'))
                if '双' in pattern:
                    weeks = [i for i in range(start, end + 1) if i % 2 == 0]
                elif '单' in pattern:
                    weeks = [i for i in range(start, end + 1) if i % 2 != 0]
            else:
                # 处理普通周次，如“1-16”
                start, end = map(int, week_str.split('-'))
                weeks = list(range(start, end + 1))
            
            for time in times:
                for week in weeks:
                    self.slots.append((week - 1, day - 1, time - 1))

    def __str__(self) -> str:
        return f"Section {self.course} - {self.time_slot}, Teacher: {self.teacher}, Rating: {self.rating}, Probability: {self.prob}"


class UserPreference:
    def __init__(self):
        self.preferred_times = [] # 喜欢的时间
        self.not_preferred_times = []  # 不喜欢的时间
        self.preferred_teachers = [] # 喜欢的老师
        self.courses_preference = [] # 课程偏好
        self.prob_preferrence = 0 # 概率偏好，越大越赌狗
    
    # 添加的method
    def add_preferred_time(self, week:int, day:int, time:int) -> None:
        if (week, day, time) not in self.preferred_times:
            self.preferred_times.append((week, day, time))

    def add_not_preferred_time(self, week:int, day:int, time:int) -> None:
        if (week, day, time) not in self.not_preferred_times:
            self.not_preferred_times.append((week, day, time))

    def add_preferred_teacher(self, teacher:str) -> None:
        if teacher not in self.preferred_teachers:
            self.preferred_teachers.append(teacher)

    def add_courses_preference(self, course:Course, weight:float) -> None:
        self.courses_preference.append((course, weight))

    def set_prob_preference(self, prob:float) -> None:
        self.prob_preferrence = prob

    def set_num_selective(self, num):
        self.num_selective = num

class Schedule:
    def __init__(self) -> None:
        self.courses = [] # 存储Course对象
        self.sections = []  # 存储Section对象
        self.slots_state = np.ones((16, 5, 14)) # 用三维数组表示该时间段是否可用，16周，5天，13节课，可用为1否则为0

    def get_schedule_features(self) -> torch.tensor:
        return torch.stack([torch.tensor(section.features, dtype=torch.float32) for section in self.sections])
    
    def get_schedule_rating(self) -> float:
        return sum(section.rating for section in self.sections)

    def is_conflict(self, new_section: Section) -> bool:
        # 检查时间冲突
        for slot in new_section.slots:
            if not self.slots_state[slot[0]][slot[1]][slot[2]]:
                return True
        return False
    
    def add_section(self, new_section: Section) -> None:
        # 如果无冲突则添加section
        if (not self.is_conflict(new_section)) and (new_section.course not in self.courses):
            self.sections.append(new_section)
            self.courses.append(new_section.course)

            # 把对应的slot状态改为不可用
            for slot in new_section.slots:
                self.slots_state[slot[0]][slot[1]][slot[2]] = 0

    # 移除section
    def remove_section(self, old_section: Section) -> None:
        if old_section in self.sections:
            self.sections.remove(old_section)
            self.courses.remove(old_section.course)

            # 把对应的slot状态改为可用
            for slot in old_section.slots:
                self.slots_state[slot[0]][slot[1]][slot[2]] = 1

    # 画图
    def draw(self,ranking) -> None:
        # 一些常数
        head = 300 # 第一行高度
        height = 400 # 其他行高度 
        left = 200 # 第一列宽度
        width = 800 # 其他列宽度
        border = 10 # 边界宽度
        white = (237, 240, 252)
        black = (0, 0, 0)
        blue = (187, 255, 255)
        green = (152, 251, 152)
        pink = (255, 192, 203)
        yellow = (250, 250, 210)
        grey = (192, 192, 192)
        colors = [blue, green, pink, yellow, grey]
        textsize1 = 80 # 字体大小
        ft1 = ImageFont.truetype("ARIALUNI.ttf", textsize1)
        img = Image.new("RGBA", (left + width*5 + border*7, head + height*13 + border * 16), black)
        draw = ImageDraw.Draw(img)
    
        draw.rectangle((border, border, border + left, border + head), fill=white)
        # 画第一列
        for row in range(1, 14):
            draw.rectangle((border, head + height*(row - 1) + border*(row + 1), border + left, head + height*row + border*(row + 1)), fill=white)
            text = f'{row}'
            # 获取文字打印大小，方便居中
            text_width = ft1.getbbox(text)[2] - ft1.getbbox(text)[0]
            text_height = ft1.getbbox(text)[3] - ft1.getbbox(text)[1]
            draw.text((border + 0.5*left - 0.5*text_width, head + height*(row - 0.5) + border*(row + 1) - text_height), text=text, fill=black, font=ft1, align='center')
        
        # 画第一行
        d = dict(zip(list(range(1, 6)), ['星期一', '星期二', '星期三', '星期四', '星期五']))
        for col in range(1, 6):
            draw.rectangle((left + border*(col + 1) + width*(col - 1), border, left + border*(col + 1) + width*col, border + head), fill=white)
            text = f'{d[col]}'
            text_width = ft1.getbbox(text)[2] - ft1.getbbox(text)[0]
            text_height = ft1.getbbox(text)[3] - ft1.getbbox(text)[1]
            draw.text((left + border*(col + 1) + width*(col - 0.5) - 0.5*text_width, border + 0.5*head - text_height), text=text, fill=black, font=ft1, align='center')
        
        # 画格子
        for row in range(1, 14):
            for col in range(1, 6):
                #if self.slots_state[0][col - 1][row - 1] == 1:
                draw.rectangle((left + border*(col + 1) + width*(col - 1), head + border*(row + 1) + height*(row - 1), left + border*(col + 1) + width*col, head + border*(row + 1) + height*row), fill=white)
                #else:
                #pass

        # 写课程名称以及老师名字
        for section in self.sections:
            color = random.choice(colors)
            text = f'{section.course.name}\n {section.teacher}'
            text_name= f'{section.course.name}'
            text_width = ft1.getbbox(text_name)[2] - ft1.getbbox(text_name)[0]
            text_height = ft1.getbbox(text_name)[3] - ft1.getbbox(text_name)[1]
            for slot in section.slots:
                
                draw.rectangle((left + border*(slot[1] + 2) + width*(slot[1]), head + border*(slot[2] + 2) + height*(slot[2]), left + border*(slot[1] + 2) + width*(slot[1] + 1), head + border*(slot[2] + 2) + height*(slot[2] + 1)), fill=color)
                draw.text((left + border*(slot[1] + 2) + width*(slot[1] + 0.5) - 0.5*text_width, head + border*(slot[2] + 2) + height*(slot[2] + 0.5) - 1.2 * text_height), text=text, fill=black, font=ft1, align='center')
        img.save('.\\out\\output'+str(ranking)+'.png')

    def __str__(self) -> str:
        return "\n".join(str(section) for section in self.sections)


# Solver 类
class Solver(ABC):
    def __init__(self, courses: list, perference: UserPreference, teachers: dict) -> None:
        # 需要对课程进行拷贝，否则会改变原来的课程
        self.courses = copy.deepcopy(courses) # 未分配的课程
        self.teachers = teachers 
        self.perference = perference
        self.explored = 0 # 已经探索的节点数
        # 如果已经存在rating_module，则直接加载
        if os.path.exists('rating_module.pkl'):
            self.rating_module = torch.load('rating_module.pkl')
        else:
            # self.rating_module = RatingModule(5)
            self.rating_module = RatingModule(4)
        
        self.rating_module.set_initial_weight(torch.tensor([1, 2, 2, 1], dtype=torch.float32))
    
    def get_explored(self) -> int:
        return self.explored
    
    @staticmethod
    def get_unassigned_courses(courses: list, schedule: Schedule) -> list: 
        # 获取未分配的课程
        unassigned_courses = []
        for course in courses:
            if course not in schedule.courses:
                unassigned_courses.append(course)
        return unassigned_courses

    @staticmethod
    def is_conflict(section1: Section, section2: Section) -> bool:
        # 检查时间冲突
        for slot in section1.slots:
            if slot in section2.slots:
                return True
        return False
    
    @staticmethod
    def select_unassigned_course_MRV(courses: list) -> Course:
        """
        Minimum Remaining Values (MRV) heuristic
        """
        # return random.choice(courses)
        return min(courses, key=lambda course: len(course.sections))

    @staticmethod
    def select_unassigned_course_random(courses: list) -> Course:
        """
        Degree heuristic
        """
        return random.choice(courses)
    
    @abstractmethod
    def solve(self, schedule: Schedule) -> Schedule:
        pass

    def preprocess_original(self, schedule: Schedule) -> None:
        Sections = []
        for course in self.courses:
            Sections += course.sections
        # 把用户不喜欢的时间状态设为不可用
        if self.perference.not_preferred_times:
            for time in self.perference.not_preferred_times:
                    schedule.slots_state[time[0]][time[1]][time[2]] = 0

        # 把用户喜欢的老师的rating变成5
        if self.perference.preferred_teachers:
            for teacher in self.perference.preferred_teachers:
                for section in self.teachers[teacher]:
                    section.rating = 5

        # 用户喜欢的时间
        if self.perference.preferred_times:
            for section in Sections:
                if set(section.slots) & set(self.perference.preferred_times):
                    section.rating = section.rating * 1.5

        # 用户对不同课的偏好
        if self.perference.courses_preference:
            for (course, weight) in self.perference.courses_preference:
                for section in course.sections:
                    section.rating = section.rating * weight
        # 处理概率
        for section in Sections:
            section.rating = section.rating * self.perference.prob_preferrence + section.rating * section.prob * (1 - self.perference.prob_preferrence)

    def preprocess(self, schedule: Schedule) -> None:
        Sections = []
        for course in self.courses:
            Sections += course.sections
        # 把用户不喜欢的时间状态设为不可用
        if self.perference.not_preferred_times:
            for time in self.perference.not_preferred_times:
                    schedule.slots_state[time[0]][time[1]][time[2]] = 0
        
        # 接下来的操作都要在特征向量上进行
        if self.perference.preferred_teachers:
            for teacher in self.perference.preferred_teachers:
                for section_c in Sections:
                    for section_t in self.teachers[teacher]:
                        if section_c.teacher == section_t.teacher and section_c.time_slot == section_t.time_slot:
                            section_c.features[1] = 2

                # for section in self.teachers[teacher]:
                #     print('finish adding one teacher!')
                #     section.features[1] = 1

        # 用户喜欢的时间 
        if self.perference.preferred_times:
            for section in Sections:
                if set(section.slots) & set(self.perference.preferred_times):
                    section.features[2] = 1  

        # 用户对不同课的偏好
        if self.perference.courses_preference:
            for (course, weight) in self.perference.courses_preference:
                for section in Sections:
                    section.features[3] = weight 

        # # 处理概率: 选课社区评分 * 概率偏好 + 选课社区评分 * 选上的概率 * (1 - 概率偏好)
        for section in Sections:
            section.features[0] = section.features[0] * self.perference.prob_preferrence + section.features[0] * section.prob * (1 - self.perference.prob_preferrence)
            section.rating = self.rating_module(torch.tensor(section.features, dtype=torch.float32)).item()

    def update_model(self, rewards: list) -> None:
        """
        rewards：反馈指的是用户对于我们找到的最优n个解的排序，如果用户与我们的偏好刚好相反，则应该为[3, 2, 1]
        """
        # 数据准备
        n = len(rewards)
        # 如果rewards 与现在的rating排序相同，则直接跳过
        if rewards == [i for i in range(1, n+1)]: 
            return
        
        self.schedule_list.sort(key=lambda schedule: schedule.get_schedule_rating(), reverse=True)
        ratings = [schedule.get_schedule_rating() for schedule in self.schedule_list[:n]]
        labels = [ratings[i-1] for i in rewards]## (51, 59, 49)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # schedule.get_schedule_features()返回的是tensor，X的每个元素是一个tensor
        X = [schedule.get_schedule_features() for schedule in self.schedule_list[:n]]
        X = torch.stack(X)
        # X = torch.stack(X)
        # 模型训练
        course_num = len(self.courses)
        # print(course_num)
        # print(X[0].shape)
        # print(len(X[0]))
        model = ScheduleRating(self.rating_module, course_num)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练两次
        model.train()
        for _ in range(2):
            for features, label in zip(X, labels):
                optimizer.zero_grad() #清零梯度
                output = model(features) #前向传播
                print('output:', output)
                print('label:', label)
                # output = torch.tensor(output, dtype=torch.float32)
                loss = criterion(output, label) #计算损失
                loss.backward() #反向传播
                optimizer.step()
            # optimizer.zero_grad() #清零梯度
            # output = model(X) #前向传播
            # loss = criterion(output, labels)
            # loss.backward() #反向传播
            # optimizer.step()
        # 更新rating_module
        self.rating_module = model.rating_module
        # 保存rating_module
        torch.save(self.rating_module, 'rating_module.pkl')


    @staticmethod
    def forward_checking(unassigned_courses: list, schedule: Schedule) -> bool:
        """
        前向检查，返回布尔值，如果为False，则说明无解
        """
        for unassigned_course in unassigned_courses:
            removed_sections = []
            for section in unassigned_course.sections:
                if schedule.is_conflict(section) and section not in removed_sections:
                    removed_sections.append(section)
            # Remove sections outside the loop
            for section in removed_sections:
                unassigned_course.sections.remove(section)
            if len(unassigned_course.sections) == 0:
                return False
        return True

    @staticmethod
    def AC3(unassigned_courses: list) -> bool:
        """
        执行弧一致性算法，返回布尔值，如果为False，则说明无解
        """
        def revise(course1: Course, course2: Course):
            """Return whether we revised the domain of course1.
            """
            revised = False

            sections_to_remove = []
            for section1 in course1.sections:
                bool_list = [] # 用来存储是否有section1和course2的section冲突
                for section2 in course2.sections:
                    if Solver.is_conflict(section1, section2):
                        bool_list.append(True)
                    else:
                        bool_list.append(False)
                if all(bool_list):
                    if len(bool_list) == 0:
                        print('error!!!')
                    sections_to_remove.append(section1)
                    revised = True
            # Remove sections outside the loop
            for section in sections_to_remove:
                # print('remove section!!!')
                course1.sections.remove(section)
            return revised
        
        # def revise_with_section(course1: Course, section: Section):
        #     """Return whether we revised the domain of course1.
        #     """
        #     revised = False

        #     sections_to_remove = []
        #     for section1 in course1.sections:
        #         if Solver.is_conflict(section1, section) and section1 not in sections_to_remove:
        #             sections_to_remove.append(section1)
        #             revised = True
        #     # Remove sections outside the loop
        #     for section in sections_to_remove:
        #         course1.sections.remove(section)
        #     return revised
        # 用一个队列来存储所有的未分配课程和未分配课程的组合
        if len(unassigned_courses) == 1:
            return True

        queue = [(course1, course2) for course1 in unassigned_courses for course2 in unassigned_courses if course1 != course2]
        while queue:
            course1, course2 = queue.pop(0)
            if revise(course1, course2):
                if len(course1.sections) == 0:
                    return False
                for course in unassigned_courses:
                    if course != course1 and course != course2:
                        queue.append((course, course1))
        return True


class backtrack_Solver(Solver):
    """
    返回第一个可行的解
    """
    def __init__(self, courses: list, perference: UserPreference, teachers: dict):
        super().__init__(courses, perference, teachers)
    
    # 只找到一个解  
    def backtrack_with_AC_helper(self, schedule: Schedule, courses: list, flag: bool) -> Schedule:
        """
        用回溯算法排课，schedule是一个当前课程表，courses未分配的课程(定义域的变化也记录期中)
        """
        if len(schedule.courses) == len(self.courses):
            self.schedule = schedule
            return schedule
        
        # AC3算法，此时courses的定义域已经变化
        unassigned_courses = Solver.get_unassigned_courses(courses, schedule)
        if not Solver.AC3(unassigned_courses, schedule):
            # 说明此时已经无解了
            return None

        # 选择一个未分配的课程
        course = Solver.select_unassigned_course_MRV(courses) if flag else Solver.select_unassigned_course_random(courses)
        for section in course.sections:
            # 此时由于执行过AC3可以直接赋值
            new_schedule = copy.deepcopy(schedule)
            new_schedule.add_section(section)
            # 先除去，再拷贝，最后加回
            courses.remove(course)
            new_courses = copy.deepcopy(courses)
            courses.append(course)
            # new_courses = copy.deepcopy(courses)
            self.solve_helper(new_schedule, new_courses, flag)
            self.explored += 1 # 探索数加一
            result = self.naive_backtrack_with_AC_helper(new_schedule, new_courses)
            if result is not None:
                return result
        return None 
    
    def backtrack_with_AC(self, schedule: Schedule):
        courses = copy.deepcopy(self.courses)
        return self.backtrack_with_AC_helper(schedule, courses)


class optimal_solver(Solver):
    """
    找到排课问题最优解的抽象类
    """
    def __init__(self, courses: list, perference: UserPreference, teachers: dict):
        super().__init__(courses, perference, teachers)
        self.schedule_list = []
        self.best_rating = 0
        self.best_schedule = None

    def solve(self, schedule: Schedule, n: int = 1, flag: bool = False) -> Schedule:
        """
        @schedule: 一个空的课程表
        @n: 返回最优的n个解
        @flag: 用来判断是否采用MRV启发式选择未分配课程，如果为True，则采用MRV启发式选择未分配课程，否则采用随机选择
        """
        self.preprocess(schedule)
        # 在预处理之后，courses的定义域已经变化，
        courses = copy.deepcopy(self.courses)
        if flag:
            # 把课程按照定义域的大小排序,从小到大
            courses.sort(key=lambda course: len(course.sections))
        else:
            # 随机打乱课程
            random.seed(5201314)
            random.shuffle(courses)

        self.solve_helper(schedule, courses)
        self.schedule_list.sort(key=lambda schedule: schedule.get_schedule_rating(), reverse=True)
        return self.schedule_list[:n]
    
    @abstractmethod
    def solve_helper(self, schedule: Schedule, courses: list) -> None:
        
        raise NotImplementedError


class optimal_naive_Solver(optimal_solver):
    """
    暴力搜索，不采用弧一致性，返回所有可行解中最优的解
    """
    def __init__(self, courses: list, perference: UserPreference, teachers: dict):
        super().__init__(courses, perference, teachers)

    def solve_helper(self, schedule: Schedule, courses: list) -> None:
        if len(schedule.courses) == len(self.courses):
            self.schedule_list.append(schedule)
            return None
        
        course = courses[0]

        for section in course.sections:
            self.explored += 1
            if schedule.is_conflict(section):
                continue
            new_schedule = copy.deepcopy(schedule)
            new_schedule.add_section(section)
            # 先除去，再拷贝，最后加回
            courses.remove(course)
            new_courses = copy.deepcopy(courses)
            # courses.append(course)
            # 加到第一个位置
            courses.insert(0, course)
            # new_courses = copy.deepcopy(courses)
            self.solve_helper(new_schedule, new_courses)
        return None

class optimal_forward_checking_Solver(optimal_solver):
    def __init__(self, courses: list, perference: UserPreference, teachers: dict):
        super().__init__(courses, perference, teachers)
    
    def solve_helper(self, schedule: Schedule, courses: list) -> None:
        if len(schedule.courses) == len(self.courses):
            self.schedule_list.append(schedule)
            return None
    
        # unassigned_courses = Solver.get_unassigned_courses(courses, schedule)

        if not Solver.forward_checking(courses, schedule):
            return None
        
        # print(f'--{len(courses)}---{len(self.courses)}--{len(schedule.courses)}')
        # course = Solver.select_unassigned_course_MRV(courses) if flag else Solver.select_unassigned_course_random(courses)
        course = courses[0]

        for section in course.sections:
            self.explored += 1
            new_schedule = copy.deepcopy(schedule)
            new_schedule.add_section(section)
            # 先除去，再拷贝，最后加回
            courses.remove(course)
            new_courses = copy.deepcopy(courses)
            # 加到第一个位置
            courses.insert(0, course)
            # new_courses = copy.deepcopy(courses)
            self.solve_helper(new_schedule, new_courses)
        return None


class optimal_AC3_Solver(optimal_solver):
    def __init__(self, courses: list, perference: UserPreference, teachers: dict):
        super().__init__(courses, perference, teachers)


    def solve_helper(self, schedule: Schedule, courses: list) -> None:
        """
        采用弧一致性缩小算法的搜索范围
        """
        if len(schedule.courses) == len(self.courses):
            self.schedule_list.append(schedule)
            return None
        
        # unassigned_courses = Solver.get_unassigned_courses(courses, schedule)
        # 先执行forward_checking
        if not Solver.forward_checking(courses, schedule):
            return None
        
        #再执行AC3算法
        if not Solver.AC3(courses):
            return None
        
        # 选择一个未分配的课程
        # course = Solver.select_unassigned_course_MRV(courses) if flag else Solver.select_unassigned_course_random(courses)
        course = courses[0]

        for section in course.sections:
            self.explored += 1
            # 此时由于执行过AC3可以直接赋值
            new_schedule = copy.deepcopy(schedule)
            new_schedule.add_section(section)
            # 先除去，再拷贝，最后加回
            courses.remove(course)
            new_courses = copy.deepcopy(courses)
            # 加到第一个位置
            courses.insert(0, course)
            # new_courses = copy.deepcopy(courses)
            self.solve_helper(new_schedule, new_courses)

        return None
    

###########################################################################
#以下对权重模型的构建
###########################################################################
class RatingModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def set_initial_weight(self, initial_weight):
        if initial_weight.nelement() == self.linear.weight.nelement():
            self.linear.weight.data = initial_weight.view_as(self.linear.weight)
            self.linear.weight.requires_grad = True
        else:
            raise ValueError("Initial weight tensor has an incorrect size")

    def forward(self, x):
        # print(x)
        return self.linear(x)

class ScheduleRating(nn.Module):
    def __init__(self, rating_module, course_num):
        super().__init__()
        self.course_num = course_num
        self.rating_module = rating_module
    
    def forward(self, x):
        # 确保 x 是一个包含所有课程特征的列表
        if not isinstance(x, torch.Tensor) or x.size(0) != self.course_num:
            raise ValueError(f"Expected {self.course_num} course features, got {x.size(0)}")

        outputs = []
        for i in range(self.course_num):
            outputs.append(self.rating_module(x[i]))

        return sum(outputs).squeeze()
    
# # example
# input_dim = 5
# course_num = 5
# rating_module = RatingModule(input_dim)
# rating_module.set_initial_weight(torch.tensor([1, -1, 0.5, 0.5, 0.5], dtype=torch.float32))
# model = ScheduleRating(course_num, rating_module)

# cirterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)


    