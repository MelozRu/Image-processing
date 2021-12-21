# Для работы с json файлами
import json

# Расчет онлайн стоимости обчучения за год
# Принимает 
# jobj - файл типа json с данными
# teach - кол-во преподователей
# stud - кол-во студентов
# group - кол-во групп
def calc_online_year(jobj, teach, stud, group):
    # Получаем данные из файла по ключу
    teachers_salary = teach * jobj['teachers_salary']
    admin_salary = jobj['admin_salary']
    accountants_salary = jobj['accountants_salary']
    methodist_salary = 2 * jobj['methodist_salary']
    internet = jobj['internet']
    c1 = jobj['1c']
    
    # Высчитываем стоимость за год обучения на одного студента
    payment = 12 * (teachers_salary + admin_salary 
                    + accountants_salary + methodist_salary 
                        + internet + c1) / (stud * group)
    return payment

# Расчет офлайн стоимости обчучения за год
# Принимает 
# jobj - файл типа json с данными
# teach - кол-во преподователей
# stud - кол-во студентов
# group - кол-во групп
def calc_offline_year(jobj, teach, stud, group):
    # Получаем данные из файла по ключу
    teachers_salary = teach * jobj['teachers_salary']
    admin_salary = jobj['admin_salary']
    accountants_salary = jobj['accountants_salary']
    methodist_salary = 2 * jobj['methodist_salary']
    wardrobe_salary = 1 * jobj['wardrobe_salary']
    security_guard_salary = 1 * jobj['security_guard_salary']
    square_meter = stud * 5 * jobj['square_meter']
    utilities = stud * 5 * jobj['utilities']
    c1 = jobj['1c']   
    
    # Высчитываем стоимость за год обучения на одного студента
    payment = 12 * (teachers_salary + admin_salary 
                    + accountants_salary + methodist_salary 
                        + wardrobe_salary + security_guard_salary
                        + square_meter + utilities 
                        + c1) / (stud * group)
    return payment

# функция расчёта стоимости обучения
# Принимает 
# type_study - тип обучения (online or offline)
# num_of_group - кол-во групп на потоке
# num_of_stud - кол-во студентов в группе
# num_of_pred - кол-во предметов в месяце
def payment(type_study, num_of_group, num_of_stud, num_of_pred):
    
    # открываем файл json и читаем от туда данные, получаем данные в виде текста
    fin = open('./data/{}.json'.format(type_study), encoding='utf8')
    jsonText = fin.read()
    fin.close()

    # Преобразуем текст в справочник
    jsonObj = json.loads(jsonText)

    # Считаем количество преподователей
    num_of_teach = num_of_pred / 3

    # определяем тип обучения и вызываем соответствуюшюю функцию
    if type_study == 'online':
        student_payment = calc_online_year(jsonObj, num_of_teach, num_of_stud, num_of_group)
    elif type_study == 'offline':
        student_payment = calc_offline_year(jsonObj, num_of_teach, num_of_stud, num_of_group)
    else:
        # Случай не верно указанного типа обучения
        print('Incorrect type of training (online or offline)')

    return student_payment

def main():
    # Расчитываем стоимость обучения online и offline
    online = payment('online', 4, 22, 72)
    offline = payment('offline', 4, 22, 72)

    # Выводим результат
    print('The student must pay if studying online - ', online)
    print('The student must pay if studying offline - ', offline)

# Вызов главной функции main
if __name__ == "__main__":
    main()