## Загрузка вспомогательных библиотек
import os
import pandas as pd
import datetime as dt
import missingno as msno
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np



## Подготовка вспомогательных функций, используемых в исследовании


### Функция загрузки датасета
def file_load(file_name):
    """
    Функция загружает Датасет из файла источника

        Параметры:
            file_name (str): имя файла с данными
        Выходные параметры (DataFrame)

    """
    file_path = '../Data/' + file_name
    print('... загружаем файл с данными датасета ...')
    df = pd.read_csv(file_path)
    file_info(df, file_path)
    print('\n... готовим иллюстрацию заполненности датасета значениями ...')
    msno.matrix(df)  # визуализируем заполнение занчениями датасета

    return df


### Функция определения характеристик файла источника данных
def file_info(df, file_path):
    """
    Функция описывает файл источник для загруженного датасета

        Параметры:
            df (DataFrame): загруженный датасет
            file_path (sgr): отрносительнрый путь к файлу с датасетом
        Выходные параметры (None)

    """
    print(f'Источником данных является файл: {file_path[8:]}'
          f'\nРазмер файла {os.stat(file_path).st_size} байт'
          f'\nВ файле содержится 1 таблица:'
          f'\n  - количество строк: {df.shape[0]}'
          f'\n  - количество столбцов: {df.shape[1]}')


### Функция коррекции данных в датасете
def dataset_preparation(df):
    """
    Функция исправляет типы данных, убирает пропуски и дубликаты в данных датасета

        Параметры:
            df (DataFrame): корректируемый датасет
        Выходные параметры (DataFrame)

    """
    print('\nЭтап 1. ... запускаем проверку идентичности типов данных в полях датасета ...')
    noncorrect_columns = checking_type_error(df)
    print('... запускаем коррекцию типов данных в полях датасета ...')
    correct_type(df, noncorrect_columns)

    print('\nЭтап 2. ... запускаем замену некорректного указания типа None ...')
    df = df.replace(['nan'], np.nan)
    print('Все некорректные указания на пустые значеия заменены на None.')

    print('\nЭтап 3. ... запускаем изменение типов данных в полях даты и времени ...')
    if 'visit_date' in df.columns:
        df['visit_date'] = df.visit_date.apply(lambda x:
                                               dt.datetime.strptime(x, '%Y-%m-%d').date())
        print("В поле 'visit_date' тип данных изменён на datetime.")
    if 'visit_time' in df.columns:
        df['visit_time'] = df.visit_time.apply(lambda x:
                                               dt.datetime.strptime(x, '%H:%M:%S').time())
        print("В поле 'visit_time' тип данных изменён на datetime.")
    if 'hit_date' in df.columns:
        df['hit_date'] = df.hit_date.apply(lambda x:
                                           dt.datetime.strptime(x, '%Y-%m-%d').date())
        print("В поле 'hit_date' тип данных изменён на datetime.")

    print('\nЭтап 4. ... анализируем пропущенные значения в датасете ...')
    data_set_audit(df)

    if 'visit_date' in df.columns:
        # Список удаляемых колонок
        # (в колонках более 40% пропущенных данных)
        columns_columns_delete = ['utm_keyword', 'device_os', 'device_model']

        # Список колонок, в которых проверется наличие пропусков для удаления строк
        # (в колонках менее 1% пропущенных данных)
        columns_rows_delete = ['utm_source']

        # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
        # (в колонках более 1% и менее 40% пропущенных данных)
        columns_value_top = ['utm_campaign', 'utm_adcontent', 'device_brand']

        # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
        # при этом одновременно учитываются корреляциии атрибута с другим атрибутом
        # (в колонках более 1% и менее 40% пропущенных данных)
        # columns_value_top_correlation = [('geo_country', 'geo_city')]

    #     else:
    #         # Список удаляемых колонок
    #         # (в колонках более 40% пропущенных данных)
    #         columns_columns_delete = ['hit_referer', 'event_value']

    #         # Список колонок, в которых проверется наличие пропусков для удаления строк
    #         # (в колонках менее 1% пропущенных данных)
    #         columns_rows_delete = ['hit_number', 'hit_type', 'hit_page_path',\
    #                        'event_category', 'event_action']

    #         # Список колонок, в которых меняются пропуски на наиболее часто встречающиеся значения
    #         # (в колонках более 1% и менее 40% пропущенных данных)
    #         columns_value_top = ['device_brand', 'event_label']

    # Удаляем пропуски в датасете

    df = clean_columns_rows(df, columns_columns_delete,
                            columns_rows_delete,
                            columns_value_top)

    # Удаляем дубликаты в датасете
    print('\nЭтап 5. ... запускаем удаление дубликатов ...')
    df = df.drop_duplicates()
    print('Дубликаты записей удалены.')
    print('Размер Датасета после удаление дубликатов:', df.shape)

    if 'visit_date' in df.columns:
        print('\nЭтап 6. ... запускаем поиск и удаление аномалий ...')
        df = delete_anomalies(df)
        print('Записи с клиентами, имеющими аномально большое кодичество визитов - удалены.')
        print(f'Размер Датасета после удаления аномалий: {df.shape}')


    return df


### Функция проверки всех колонок датасета на единство типа данных в колонке
def checking_type_error(df):
    """
    Функция проверяет каждое поле датасета на несовпадение типов данных и выводит поля с указанием количества записей разных типов

        Параметры:
            df (DataFrame): проверяемый датасет
        Выходные параметры:
            columns_noncorrect_types (list): список полей датасета, в каждом из которых найдены разные типы данных

    """
    flagCorrect = True
    columns_noncorrect_types = []
    for elem in df.columns:
        df_type = df[elem].apply(lambda x: type(x)).value_counts().to_frame()
        if len(df_type) > 1:
            columns_noncorrect_types.append(elem)
            if flagCorrect:
                print('Для одного атрибута обнаружены данные разного типа в следующих полях')
                print(' ------------------------------------------------------------ ')
                print('|  НАИМЕНОВАНИЕ ПОЛЯ  |  ТИП ДАННЫХ  |   КОЛИЧЕСТВО ЗАПИСЕЙ  |')
                print(' ------------------------------------------------------------ ')
            flagCorrect = False

            for i in range(len(df_type)):
                print('   ', df_type.columns[0], ' ' * (21 - len(df_type.columns[0])),
                      str(df_type.index[i])[7:-1], ' ' * (21 - len(str(df_type.index[i])[7:-1])),
                      str(df_type[df_type.columns[0]].values[i]))
            print(' ------------------------------------------------------------ ')
    if flagCorrect:
        print('Полей с разными типами данных в одном атрибуте не обнаружено!')

    return columns_noncorrect_types


### Функция изменения типа данных с 'float' на 'str'
def correct_type(df, noncorrect_columns):
    """
    Функция изменяет типы данных на 'str' в заданных полях датасета

        Параметры:
            df (DataFrame): датасет, в котором корректируются типы данных полей
            noncorrect_columns (list): список полей датасета, в которых функция корректирует тип данных
        Выходные параметры (None)

    """
    for elem in noncorrect_columns:
        df[elem] = df[elem].apply(lambda x: str(x))
    print(f'Коррекция типов данных завершена в {len(noncorrect_columns)} полях')
    print('... запущена перепроверка идентичности типов данных ... ')
    checking_type_error(df)


### Функция проверки наличия незаполненных значений в колоноках датасета
def data_set_audit(df):
    """
    Функция проверяет пропуски в Датасете и выводит для каждого поля долю пропусков в % от общего количесвтва записей

        Параметры:
            df (DataFrame): датасет, в котором проверяются пропуски
        Выходные параметры (None)

    """
    print('Размер анализируемого Датасета:', df.shape)
    nan_columns = [(elem, df[elem].isna().sum(), type(df.loc[0, elem]))
                   for elem in df.columns
                   if df[elem].isnull().describe()[1] > 1 or
                   df[elem].isnull().describe()[1] == 1 and
                   df[elem].isnull()[0] == True]
    if len(nan_columns) > 0:
        print('ПРОПУСКИ  В  КОЛОНКАХ  ДАТАСЕТА')
        print('========================================================================')
        print('  Поле                        Пропуски             Тип данных в колонке ')
        print('                           (кол-во,      %)                             ')
        print('------------------------------------------------------------------------')
        for elem in nan_columns:
            print(' ', elem[0],
                  ' ' * (24 - len(elem[0])), elem[1],
                  ' ' * (9 - len(str(elem[1]))), round((elem[1] / len(df) * 100), 2),
                  ' ' * 12, str(elem[2])[7:-1])
        print('========================================================================')
    else:
        print('Пропущенных данных в Датасете не обнаружено!')


### Функция отчистки столбцов и строк от нулевых значений
def clean_columns_rows(df, col_col_df, col_row_df, col_cell_val_top):
    """
    Функция удаляет заданные столбы. Затем функция удаляет строки, в которых есть пропущеные данные

        Параметры:
            df (DataFrame): датасет, в котором удаляются столбы, затем проверяются пропуски и удаляют соответвующие строки
            col_col_df (list): список колонок, которые необходимо удалить.
            col_row_df (list): списко колонок, при наличии пропусках в которых - необходимо удалить строки.
            col_cell_val_top (list): список колонок, в которых необходимо замеить пропуски простым выбором наиболее часто встречающихся значений.
            col_cell_val_top_corr (list(typle)): списко кортежей, в которых указываются родительские и дочернии колонки для подбора наиболее часто встречающихся значений в дочерней колонке с учтом соответствующего значения в родительской колонке.
        Выходные параметры:
            df_clear (DataFrame): датасет, отчищенный от пропусков

    """

    print('... удаляем колонки, в которых более 40% пропущенных данных ...')
    df = df.drop(columns=col_col_df, axis=1)  # Удаление колонки 'hit_time'
    print('После удаления заданных колонок - размер датасета:', df.shape)

    print('... удаляем строки, для которых в колонках менее 1% пропущенных данных ...')
    df = df.dropna(subset=col_row_df, axis=0, how='any')  # Удаление строк, в которых выявлены пропущенные данные
    print(f'Удаление колонок и строк с нулевыми данными - завершено.')

    print('... меняем пропущенные данные на наиболее часто встречающиеся ... ')
    for col in col_cell_val_top:
        nan_indexes = df[df[col].isnull()].index
        df.loc[nan_indexes, col] = df[col].describe().loc['top']

    print('... запущена перепроверка отсутствия нулевых данных ... ')
    data_set_audit(df)

    #     print('\n')
    #     for column_elem in col_cell_val_top_corr:
    #         parent_column, child_column = column_elem[0], column_elem[1]

    #         # 1. Определяем уникальные значения родительской колонки,
    #         #    имеющие пустые значения в дочерней колонке
    #         columns_non_correct = list(df_clear[parent_column].loc[df_clear[child_column].isnull()].unique())

    #         # 2. заполняем пустоты названиями, встречающимеся чаще всего среди значений
    #         #    дочерней колонки для соответвующего значения родительской колонки Датасета

    #         for col_cor in columns_non_correct:

    #             # Массив индексов пустых значений в дочерней колонке Датасета,
    #             # соответсвующих значению родительской колонки Датасета
    #             index_nan_values = df_clear[child_column].\
    #                             loc[df_clear[child_column].isnull()].\
    #                             loc[df_clear[parent_column] == col_cor].index

    #             # Самое часто встречающееся значение в дочерней колонке,
    #             # соответсвущее значению родительской колонки Датасета
    #             top_value = str(df_clear[child_column].\
    #                             loc[df_clear[child_column].notnull()].\
    #                             loc[df_clear[parent_column] == col_cor].\
    #                             describe().top)

    #             if top_value not in [None, 'nan', 'null', 'Nan', 'NaN']:
    #                 df_clear.loc[index_nan_values, child_column] = top_value

    #         print('... в зависимых колонках меняем пропуски на самые частые значения ... ')
    #         print(f'Заменена нулевых данных, для которых выявлены наиболее часто встречающиеся значения - завершена\n')

    return df


### Функция выявления и удаления аномалий в датасете визитов
def delete_anomalies(df):
    """
    Функция определяет количество аномалий в датасете визитов и удаляет их из датасета

        Параметры:
            df (DataFrame): датасет визитов
        Выходные параметры (DataFrame)

    """
    # Вычисление количесвтва визитов каждого клиента по 'client_id'
    visit_count = df.groupby('client_id').agg('count')

    # Вычисление количесвта дней в периоде фиксации данных в датасете
    days_count = (df.visit_date.max() - df.visit_date.min()).days

    # Определение идентификаторов клиентов, у которых в среднем больше
    # одного визита в день
    roboticity_client = list(visit_count. \
                             loc[visit_count['session_id'] >= days_count].index)

    # Исключение клиентов, у которых в среднем больше одного визита в день
    # df = df.query(f'client_id not in {roboticity_client}')
    df = df.loc[~df.client_id.isin(roboticity_client)]

    return df


### Агрегация параметров датасета визитов во времени
def visual_plots(df):
    # Подготовка полей для группировки датасета
    df['client'] = df['client_id']

    # Словарь агрегирующих функций по атрибутам и агрегация для визуализации
    ag_dict = {'client_id': 'max', 'visit_date': 'min', 'visit_time': 'min',
               'geo_country': 'max', 'geo_city': 'max',
               'device_category': 'max', 'device_browser': 'max',
               'utm_source': 'max', 'utm_medium': 'max'}

    df_ag = df.groupby('client').agg(ag_dict)

    # Агрегация посещений новыми клиентами по месяцам
    df_ag['year-month'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%Y-%m"))
    ag_ym = df_ag.groupby('year-month').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по дням месяца
    df_ag['dayOFmonth'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%d"))
    ag_dm = df_ag.groupby('dayOFmonth').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по дням недели
    df_ag['dayOFweek'] = df_ag['visit_date'].apply(lambda x: dt.datetime.strftime(x, "%w"))
    ag_dw = df_ag.groupby('dayOFweek').agg({'client_id': 'count'})

    # Агрегация посещений новыми клиентами по часам суток
    df_ag['hourOFday'] = df_ag['visit_time'].apply(lambda x: x.hour)
    ag_hd = df_ag.groupby('hourOFday').agg({'client_id': 'count'})

    return [(ag_ym, 'по мясецам'),
            (ag_dm, 'по дням месяца'),
            (ag_dw, 'по дням недели'),
            (ag_hd, 'по часам дня')]


### Визуализация параметров датасета визитов во времени
def show_plot(df, title):
    fig, bx = plt.subplots(figsize=(12, 3))
    bx.plot(df.index, df['client_id'], color='green')
    bx.set_xlabel('Периоды')
    bx.set_ylabel('Количество уникальных визитов')
    bx.set_title(f'Распределение визитов {title}')
    bx.yaxis.set_major_formatter(FormatStrFormatter('%.0f'));


def fig_hist(dfs):
    for elem in dfs:
        show_plot(elem[0], elem[1])


### Функция конкатенации строки даты и времени
def create_date_time_visit(date, time):
    """
    Функция сцепляет строковое значение даты и времени визитов.
    Предназнаена для использования с датасетом визитов

        Параметры:
            date (str): дата визита
            time (str): время визита
        Выходные параметры (str, None)

    """
    if pd.notna(date) and pd.notna(time):
        return str(date) + ' ' + str(time)
    elif pd.notna(date):
        return str(date) + str(' 00:00:00')
    else:
        return None


### Функция обогощения времени визита и времени каждого события (в наносекундах)
def create_date_time_ns(date_time, date, ns):
    """
    Функция обогащает время визита при его отсутсвии в датасете визитов и сцепляет строковое значение времени событий и визитов.
    Предназнаена для использования с объединённым датасетом

        Параметры:
            date_time (str): дата и время визита
            data (str): дата события
            ms (str): время (в наносекундах)
        Выходные параметры (str, None)

    """
    if pd.notna(date_time) and pd.notna(ns):
        return str(date_time) + f'{int(ns) / 1000000000:.9f}'[1:]
    elif pd.notna(date_time):
        return str(date_time) + '.000000000'
    elif pd.isna(date_time) and pd.notna(date) and pd.notna(ns):
        return str(date) + f' 00:00:0{int(ns) / 1000000000:.9f}'
    elif pd.isna(date_time) and pd.notna(date) and pd.isna(ns):
        return str(date) + ' 00:00:00.000000000'
    else:
        return None


### Функция объединения датасетов
def data_marge(df_pk, df_fk, key):
    """
    Функция объединяет два датасета и иллюстрирует пропуски данных

        Параметры:
            df_pk (DataFrame): датасет, с первичным ключом, используемым для объединени
            df_fk (DataFrame): датасет, с внешним ключом, используемым для объединения
        Выходные параметры:
            df (DataFrame): объединённый датасет

    """
    print(f"\n... объединяем датасеты по ключу '{key}' ...")
    df = df_pk.merge(df_fk, left_on=key, right_on=key, how='outer')
    print('Датасеты объединёны.')
    print('... очищаем память от загруженных ранее промежуточных датасетов ...')
    del df_pk, df_fk  # очищаем память от ненужных датасетов
    print('Промежуточные датасеты удалены.\n')

    print('\n... запускаем объединение полей даты и времени события ...')
    df['date_time'] = df.apply(lambda x:
                               create_date_time_ns(x.date_time, x.hit_date, x.hit_time),
                               axis=1)
    print("Объединение полей даты и времени событий осуществлено в поле 'date_time'.")

    print("\n... запускаем преобразование типа в полей 'date_time' ...")
    df['date_time'] = pd.to_datetime(df['date_time'])
    print("Преобразование типа в поле 'date_time' на тип `datetime64(ns)` завершено .")

    df.drop(columns=['hit_date', 'hit_time'], inplace=True)
    print("Поля 'hit_date' и 'hit_time' удалены из объединённого датасета.")
    size_df = df.shape
    print(f'Датасет содержит {size_df[0]} строк и {size_df[1]} столбцов.')

    print('\n... анализируем пропущенные значения в датасете ...')
    data_set_audit(df)  # запускаем проверку на пустые значения в датасете
    print('Анализ пропущенных значений в датасете завершён.\n')

    print('... готовим иллюстрацию заполненности датасета значениями ...')
    msno.matrix(df)  # визуализируем заполнение занчениями датасета

    return df