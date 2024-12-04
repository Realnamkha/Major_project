from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from django.conf import settings

POPULATION_SIZE = 300
NUMB_OF_ELITE_SCHEDULES = 10
TOURNAMENT_SELECTION_SIZE = 5
MUTATION_RATE = 0.01
VARS = {'generationNum': 0,
        'terminateGens': False}


fitness_values = []

class Population:
    def __init__(self, size):
        self._size = size
        # self._data = data
        self._schedules = [Schedule().initialize() for i in range(size)]

    def getSchedules(self):
        return self._schedules
    
    def __str__(self):
        # Create a string representation of the Population
        schedule_descriptions = [str(schedule) for schedule in self._schedules]
        return (f"Population Size: {self._size}\n"
                f"Schedules: \n" + "\n".join(schedule_descriptions))
    
    


class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()
        self._sections = Section.objects.all()

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_sections(self):
        return self._sections


class Class:
    def __init__(self, dept, section, course):
        self.department = dept
        self.course = course
        self.instructor = None
        self.meeting_time = None
        self.room = None
        self.section = section

    def get_id(self):
        return self.section_id # see this later

    def get_dept(self):
        return self.department

    def get_course(self):
        return self.course

    def get_instructor(self):
        return self.instructor

    def get_meetingTime(self):
        return self.meeting_time

    def get_room(self):
        return self.room

    def set_instructor(self, instructor):
        self.instructor = instructor

    def set_meetingTime(self, meetingTime):
        self.meeting_time = meetingTime

    def set_room(self, room):
        self.room = room
        
    def __str__(self):
        return f"Class(dept={self.department}, course={self.course}, section={self.section}, instructor={self.instructor}, meeting_time={self.meeting_time}, room={self.room})"



class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._isFitnessChanged = True

    def getClasses(self):
        self._isFitnessChanged = True
        return self._classes

    def getNumbOfConflicts(self):
        return self._numberOfConflicts

    def getFitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculateFitness()
            self._isFitnessChanged = False
        return self._fitness

    def addCourse(self, data, course, courses, dept, section):
        newClass = Class(dept, section.section_id, course)

        newClass.set_meetingTime(
            data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])

        newClass.set_room(
            data.get_rooms()[random.randrange(0, len(data.get_rooms()))])

        crs_inst = course.instructors.all()
        newClass.set_instructor(
            crs_inst[random.randrange(0, len(crs_inst))])

        self._classes.append(newClass)

    def initialize(self):
        sections = Section.objects.all()
        # print(f"Total sections found: {sections.count()}")  # Debug: Number of sections

        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            # print(f"Processing section: {section} (Department: {dept}) with {n} classes per week.")

            available_meeting_times = len(data.get_meetingTimes())
            # print(f"Available meeting times: {available_meeting_times}")

            if n > available_meeting_times:
                # print(f"Reducing n from {n} to available meeting times {available_meeting_times}.")
                n = available_meeting_times  # Ensure we don't exceed available meeting times

            courses = dept.courses.all()
            # print(f"Number of courses available in department '{dept}': {courses.count()}")

            # Calculate how many classes to add
            classes_to_add = n // len(courses)
            # print(f"Adding {classes_to_add} classes for each of {courses.count()} courses.")

            for course in courses:
                for i in range(classes_to_add):
                    self.addCourse(data, course, courses, dept, section)

            additional_classes = n % len(courses)
            # print(f"Adding {additional_classes} additional classes for random courses.")

            for course in courses.order_by('?')[:additional_classes]:
                self.addCourse(data, course, courses, dept, section)

            total_classes = len(self._classes)  # Count total classes added for this section
            # print(f"Total classes added for section {section}: {total_classes}")
            


        # print(f"Finished initializing schedules. Total classes in this population: {len(self._classes)}")
        return self
    
    def parse_time(self, time_str):
        """Convert a time string (e.g., '11:30') to a time object."""
        return datetime.strptime(time_str.strip(), '%H:%M').time()


    def calculateFitness(self):
        self._numberOfHardConflicts = 0  # Track hard conflicts
        self._numberOfSoftConflicts = 0  # Track soft conflicts
        classes = self.getClasses()

        # Define weights for hard and soft constraints
        hard_weights = {
            'seating_capacity': 1,            
            'same_course_same_section': 1.5,    
            'instructor_conflict': 2,         
            'duplicate_time_section': 2,      
            'instructor_availability': 2,     
        }

        soft_weights = {
            'no_consecutive_classes': 0.5,      
            'morning_classes': 1,           
            'break_time_conflict': 0.3,         
            'balanced_days': 0.3              
        }

        # Check hard constraints
        self.check_seating_capacity(classes, hard_weights)
        for i in range(len(classes)):
            self.check_course_conflicts(classes, i, hard_weights)
            self.check_instructor_conflict(classes, i, hard_weights)
            self.check_duplicate_time(classes, i, hard_weights)
            self.check_instructor_availability(classes, i, hard_weights)

        # Check soft constraints
        for i in range(len(classes)):
            self.check_consecutive_classes(classes, i, soft_weights)
            self.check_morning_classes(classes, i, soft_weights)
            self.check_break_time_conflict(classes, i, soft_weights)

        # Check balance across the week (this could be a soft constraint as well)
        self.check_balanced_days(classes, soft_weights)

        # Fitness calculation: Combine hard and soft conflicts with different penalties
        total_conflict = (self._numberOfHardConflicts * 10) + self._numberOfSoftConflicts  # Hard conflicts are weighted more
        return 1 / (total_conflict + 1)  # Avoid division by zero

    # Check methods for hard constraints
    def check_seating_capacity(self, classes, weights):
        for i in range(len(classes)):
            if classes[i].room.seating_capacity < int(classes[i].course.max_numb_students):
                self._numberOfHardConflicts += weights['seating_capacity']

    def check_course_conflicts(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            day_i = str(classes[i].meeting_time).split()[0]
            day_j = str(classes[j].meeting_time).split()[0]
            if (classes[i].course.course_name == classes[j].course.course_name and 
                day_i == day_j and classes[i].section == classes[j].section):
                self._numberOfHardConflicts += weights['same_course_same_section']

    def check_instructor_conflict(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section != classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time and 
                classes[i].instructor == classes[j].instructor):
                self._numberOfHardConflicts += weights['instructor_conflict']

    def check_duplicate_time(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section == classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time):
                self._numberOfHardConflicts += weights['duplicate_time_section']

    def check_instructor_availability(self, classes, i, weights):
        instructor = classes[i].instructor
        availability_start = instructor.availability_start
        availability_end = instructor.availability_end
        meeting_time_str = classes[i].meeting_time.time
        start_time_str, end_time_str = meeting_time_str.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time = self.parse_time(end_time_str)

        if start_time < availability_start or end_time > availability_end:
            self._numberOfHardConflicts += weights['instructor_availability']

    # Check methods for soft constraints
    def check_consecutive_classes(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if classes[i].instructor == classes[j].instructor:
                time_i_end = self.parse_time(classes[i].meeting_time.time.split(' - ')[1])
                time_j_start = self.parse_time(classes[j].meeting_time.time.split(' - ')[0])
                if time_i_end == time_j_start:
                    self._numberOfSoftConflicts += weights['no_consecutive_classes']

    def check_morning_classes(self, classes, i, weights):
        morning_start = self.parse_time('06:00')
        morning_end = self.parse_time('12:00')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)

        if morning_start <= start_time <= morning_end:
            self._numberOfSoftConflicts += weights['morning_classes'] * 0.5  # Lower penalty for morning classes
        else:
            self._numberOfSoftConflicts += weights['morning_classes'] * 3.5  # Higher penalty for non-morning classes

    def check_break_time_conflict(self, classes, i, weights):
        break_start = self.parse_time('10:00')
        break_end = self.parse_time('10:50')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time_str, _ = classes[i].meeting_time.time.split(' - ')
        end_time = self.parse_time(end_time_str)

        if start_time < break_end and end_time > break_start:
            self._numberOfSoftConflicts += weights['break_time_conflict']

    def check_balanced_days(self, classes, weights):
        day_class_count = {}
        for cls in classes:
            day = str(cls.meeting_time).split()[0]
            if day not in day_class_count:
                day_class_count[day] = 0
            day_class_count[day] += 1
        max_day = max(day_class_count.values())
        min_day = min(day_class_count.values())
        if max_day - min_day > 2:
            self._numberOfSoftConflicts += weights['balanced_days']



        
    def __str__(self):
        """Return a string representation of the schedule."""
        classes_info = [f"{cls.course.course_name} - {cls.meeting_time} - {cls.room}" for cls in self._classes]
        return f"Schedule with {len(self._classes)} classes:\n" + "\n".join(classes_info)



class GeneticAlgorithm:
    def evolve(self, population):
        return self._mutatePopulation(self._crossoverPopulation(population))

    def _crossoverPopulation(self, popula):
        crossoverPopula = Population(0)
        for i in range(NUMB_OF_ELITE_SCHEDULES):#how is elite schedule is generated
            crossoverPopula.getSchedules().append(popula.getSchedules()[i])

        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            scheduleX = self._tournamentPopulation(popula)
            scheduleY = self._tournamentPopulation(popula)

            crossoverPopula.getSchedules().append(
                self._crossoverSchedule(scheduleX, scheduleY))

        return crossoverPopula

    def _mutatePopulation(self, population):
        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            self._mutateSchedule(population.getSchedules()[i])
        return population

    def _crossoverSchedule(self, scheduleX, scheduleY):
        crossoverSchedule = Schedule().initialize()
        for i in range(0, len(crossoverSchedule.getClasses())):
            if random.random() > 0.5:
                crossoverSchedule.getClasses()[i] = scheduleX.getClasses()[i]
            else:
                crossoverSchedule.getClasses()[i] = scheduleY.getClasses()[i]
        return crossoverSchedule

    def _mutateSchedule(self, mutateSchedule):
        schedule = Schedule().initialize()
        for i in range(len(mutateSchedule.getClasses())):
            if MUTATION_RATE > random.random():
                mutateSchedule.getClasses()[i] = schedule.getClasses()[i]
        return mutateSchedule

    def _tournamentPopulation(self, popula):
        tournamentPopula = Population(0)

        for i in range(0, TOURNAMENT_SELECTION_SIZE):
            tournamentPopula.getSchedules().append(
                popula.getSchedules()[random.randrange(0, POPULATION_SIZE)])

        # tournamentPopula.getSchedules().sort(key=lambda x: x.getFitness(),reverse=True)
        # return tournamentPopula
        return max(tournamentPopula.getSchedules(), key=lambda x: x.getFitness())



def context_manager(schedule):
    classes = schedule.getClasses()
    context = []
    for i in range(len(classes)):
        clas = {}
        clas['section'] = classes[i].section_id
        clas['dept'] = classes[i].department.dept_name
        clas['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number} {classes[i].course.max_numb_students})'
        clas['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        clas['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        clas['meeting_time'] = [
            classes[i].meeting_time.pid,
            classes[i].meeting_time.day,
            classes[i].meeting_time.time
        ]
        context.append(clas)
    return context


def apiGenNum(request):
    return JsonResponse({'genNum': VARS['generationNum']})

def apiterminateGens(request):
    VARS['terminateGens'] = True
    return redirect('home')





    



@login_required
def timetable(request):
    global data
    data = Data()
    population = Population(POPULATION_SIZE)
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False
    population.getSchedules().sort(key=lambda x: x.getFitness(), reverse=True)
    geneticAlgorithm = GeneticAlgorithm()
    schedule = population.getSchedules()[0]

    while (schedule.getFitness() != 1.0) and (VARS['generationNum'] <= 500):
        if VARS['terminateGens']:
            return HttpResponse('')

        population = geneticAlgorithm.evolve(population)
        population.getSchedules().sort(key=lambda x: x.getFitness(), reverse=True)
        schedule = population.getSchedules()[0]
        # print(schedule)
        VARS['generationNum'] += 1
        # for c in schedule.getClasses():
        #     print(c.course.course_name, c.meeting_time)
        fitness_values.append(schedule.getFitness())
        print(f'\n> Generation #{VARS["generationNum"]}, Fitness: {schedule.getFitness()}')

    plt.figure(figsize=(10, 5))  # Optional: Set figure size for better visualization
    plt.plot(range(len(fitness_values)), fitness_values)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.grid(True)

    # Save the plot as an image file
    plot_path = os.path.join(settings.MEDIA_ROOT, 'fitness_plot_300_population.png')
    plt.savefig(plot_path)
    plt.close()
# Define the break time slot
    break_time_slot = '10:00 - 10:50'  # The break time you want to use
    week_days = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday']  # List of weekdays

    # Generate break times for all weekdays
    break_times = [(break_time_slot, day) for day in week_days]
    # print(f'Schedule: {schedule.getClasses()}')
    # print(f'Sections: {data.get_sections()}')
    # print(f'Time Slots: {TIME_SLOTS}')
    # print(f'Week Days: {DAYS_OF_WEEK}')


    return render(
        request, 'timetable.html', {
            'schedule': schedule.getClasses(),
            'sections': data.get_sections(),
            'times': data.get_meetingTimes(),
            'timeSlots': TIME_SLOTS,
            'weekDays': DAYS_OF_WEEK,
            'break_times': break_times,
        })


'''
Page Views
'''

def home(request):
    return render(request, 'index.html', {})


@login_required
def instructorAdd(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('instructorAdd')
    context = {'form': form}
    return render(request, 'instructorAdd.html', context)


@login_required
def instructorEdit(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instructorEdit.html', context)


@login_required
def instructorDelete(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('instructorEdit')


@login_required
def roomAdd(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('roomAdd')
    context = {'form': form}
    return render(request, 'roomAdd.html', context)


@login_required
def roomEdit(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'roomEdit.html', context)


@login_required
def roomDelete(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('roomEdit')


@login_required
def meetingTimeAdd(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('meetingTimeAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'meetingTimeAdd.html', context)


@login_required
def meetingTimeEdit(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'meetingTimeEdit.html', context)


@login_required
def meetingTimeDelete(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('meetingTimeEdit')


@login_required
def courseAdd(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('courseAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'courseAdd.html', context)


@login_required
def courseEdit(request):
    instructor = defaultdict(list)
    for course in Course.instructors.through.objects.all():
        course_number = course.course_id
        instructor_name = Instructor.objects.filter(
            id=course.instructor_id).values('name')[0]['name']
        instructor[course_number].append(instructor_name)

    context = {'courses': Course.objects.all(), 'instructor': instructor}
    return render(request, 'courseEdit.html', context)


@login_required
def courseDelete(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('courseEdit')


@login_required
def departmentAdd(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('departmentAdd')
    context = {'form': form}
    return render(request, 'departmentAdd.html', context)


@login_required
def departmentEdit(request):
    course = defaultdict(list)
    for dept in Department.courses.through.objects.all():
        dept_name = Department.objects.filter(
            id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(
            course_number=dept.course_id).values(
                'course_name')[0]['course_name']
        course[dept_name].append(course_name)

    context = {'departments': Department.objects.all(), 'course': course}
    return render(request, 'departmentEdit.html', context)


@login_required
def departmentDelete(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('departmentEdit')


@login_required
def sectionAdd(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('sectionAdd')
    context = {'form': form}
    return render(request, 'sectionAdd.html', context)


@login_required
def sectionEdit(request):
    context = {'sections': Section.objects.all()}
    return render(request, 'sectionEdit.html', context)


@login_required
def sectionDelete(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('sectionEdit')




'''
Error pages
'''

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})
