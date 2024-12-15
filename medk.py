"""
Лабораторная работа
Студент: [Ваше ФИО]
Группа: [Номер группы]
Вариант: 10 - Breast Cancer Dataset

Описание набора данных:
Набор данных содержит характеристики клеток молочной железы, используемые для диагностики рака.
Целевая переменная 'diagnosis' имеет два значения: M (malignant - злокачественная) и B (benign - доброкачественная).
"""



# Импортируем необходимые библиотеки из задания
import os
import io
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Добавить эту строку перед импортом pyplot
import matplotlib.pyplot as plt  # Добавляем этот импорт
from pandas_profiling import ProfileReport
from autoviz.AutoViz_Class import AutoViz_Class


# Проверка наличия файла
if not os.path.exists('breast_cancer.csv'):
    print("Ошибка: Файл breast_cancer.csv не найден в текущей директории")
    exit()

# 1. Загрузка набора данных
print("\nЗадание 1. Загрузка набора данных:")
with io.open('breast_cancer.csv', 'r') as file:
    data = file.read()

# Загружаем данные из строки в pandas DataFrame
df = pd.read_csv(io.StringIO(data))

# 2. Получение информации о наборе данных
print("\nЗадание 2. Получение информации о наборе данных:")
print("Информация о наборе данных:")
print(df.info())
print("\nПервые 5 строк данных:")
print(df.head())

# 3. Обработка пустых значений и дубликатов
print("\nЗадание 3. Обработка пустых значений и дубликатов:")
print("\nИсходные пустые значения:")
print(df.isnull().sum())

# Создаем второй набор с искусственными пропусками и дубликатами
df_modified = df.copy()
# Добавляем дубликаты
df_modified = pd.concat([df_modified, df_modified.iloc[:10]], ignore_index=True)
# Создаем пропуски
df_modified.loc[20:30, 'radius_mean'] = None
df_modified.loc[40:50, 'texture_mean'] = None

print("\nИскусственные пропуски:")
print(df_modified.isnull().sum())
print("\nКоличество дубликатов:", df_modified.duplicated().sum())

# Обработка пропусков и дубликатов
df_modified = df_modified.dropna()  # удаляем строки с пропусками
df_modified = df_modified.drop_duplicates()  # удаляем дубликаты

# 4. Конструирование признаков
print("\nЗадание 4. Конструирование признаков:")
# Первый набор - только числовые признаки
# (удалены категориальные признаки для чистоты анализа числовых характеристик)
df_numeric = df.select_dtypes(include=['float64', 'int64'])
print("\nНабор 1 - только числовые признаки:", df_numeric.columns.tolist())

# Второй набор - только средние значения
# (выбраны только средние значения как базовые характеристики без учета стандартных отклонений)
mean_columns = [col for col in df.columns if 'mean' in col]
df_means = df[mean_columns]
print("\nНабор 2 - только средние значения:", df_means.columns.tolist())

# Третий набор - только важные диагностические признаки
# (выбраны ниблее значимые признаки для диагностики на основе медицинских показателей)
diagnostic_columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
df_diagnostic = df[diagnostic_columns]
print("\nНабор 3 - важные диагностические признаки:", df_diagnostic.columns.tolist())

# 5. Генерация нового набора и объединения
print("\nЗадание 5. Генерация нового набора и объединения:")
print("\nСоздан новый набор данных из 100 случайных записей")
df_new = df.sample(n=100).copy()  # берем случайные 100 записей
df_new['new_feature'] = range(len(df_new))  # добавляем новый признак

# Все виды объединений
merged_inner = pd.merge(df, df_new, on='id', how='inner')
merged_left = pd.merge(df, df_new, on='id', how='left')
merged_right = pd.merge(df, df_new, on='id', how='right')
merged_outer = pd.merge(df, df_new, on='id', how='outer')

print("\nРезультаты объединений:")
print("-" * 50)
print(f"Inner join (только совпадающие записи): {len(merged_inner)} записей")
print(f"Left join (все записи из исходного набора): {len(merged_left)} записей")
print(f"Right join (все записи из нового набора): {len(merged_right)} записей")
print(f"Outer join (все записи из обоих наборов): {len(merged_outer)} записей")
print("-" * 50)

# 6. Группировка и агрегация
print("\nЗадание 6. Группировка и агрегация:")

# Группировка 1: Среднее по диагнозу
g1 = df.groupby('diagnosis').mean()
print("\nГруппировка 1 (среднее по диагнозу):")
print(g1)

# Группировка 2: Несколько агрегирующих функций
g2 = df.groupby('diagnosis').agg({
    'radius_mean': ['min', 'max', 'mean', 'std'],
    'texture_mean': ['min', 'max', 'mean', 'std']
})
print("\nГруппировка 2 (агрегирующие функции):")
print(g2)

# Группировка 3: Подсчет количества
g3 = df.groupby('diagnosis').size()
print("\nГруппировка 3 (количество по диагнозу):")
print(g3)

# Группировка 4: Медиана
g4 = df.groupby('diagnosis').median()
print("\nГруппировка 4 (медиана):")
print(g4)

# Группировка 5: Квантили
g5 = df.groupby('diagnosis').quantile([0.25, 0.75])
print("\nГруппировка 5 (квартили):")
print(g5)

# 7. Новые признаки
print("\nЗадание 7. Создание новых признаков:")
print("\nОписание новых признаков:")
print("1. area_perimeter_ratio - отношение площади к периметру")
print("2. texture_severity - комплексный показатель текстуры")
print("3. compactness_index - индекс компактности")
print("\nПримеры значений новых признаков (первые 5 строк):")
print("-" * 50)
df_features = df.copy()
# Площадь к периметру
df_features['area_perimeter_ratio'] = df_features['area_mean'] / df_features['perimeter_mean']
# Комплексный показатель текстуры
df_features['texture_severity'] = df_features['texture_mean'] * df_features['smoothness_mean']
# Индекс компактности
df_features['compactness_index'] = df_features['perimeter_mean']**2 / df_features['area_mean']
print(df_features[['area_perimeter_ratio', 'texture_severity', 'compactness_index']].head())
print("-" * 50)

# 8. Составной индекс
print("\nЗадание 8. Создание составного индекса:")
print("\nСоздание составного индекса из полей 'diagnosis' и 'radius_mean'")
df_indexed = df_features.copy()
df_indexed.set_index(['diagnosis', 'radius_mean'], inplace=True)
print("\nПример данных с составным индексом (первые 5 строк):")
print("-" * 50)
print(df_indexed.head())
print("-" * 50)
print("\nСтруктура составного индекса:")
print(df_indexed.index.names)

# 9. Кодирование категориальных признаков
print("\nЗадание 9. Кодирование категориальных признаков:")
print("\n1. Label Encoding (M -> 1, B -> 0):")
print("-" * 50)
# Способ 1: Label Encoding через map
df_encoded = df.copy()
df_encoded['diagnosis_coded'] = df_encoded['diagnosis'].map({'M': 1, 'B': 0})
print(df_encoded[['diagnosis', 'diagnosis_coded']].head())
print("\n2. One-Hot Encoding (разделение на бинарные столбцы):")
print("-" * 50)
# Способ 2: One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['diagnosis'])
print(df_onehot.filter(like='diagnosis').head())
print("-" * 50)

# 10. Статистические данные
print("\nЗадание 10. Основные статистические показатели:")
print("\nОбщая информация:")
print(f"Количество записей: {len(df)}")
print(f"Количество признаков: {len(df.columns)}")
print("\nСтатистика по числовым признакам:")
print("-" * 50)
print(df.describe().round(3))  # Округляем до 3 знаков после запятой
print("-" * 50)

# 11. Визуализация через pandas
print("\nЗадание 11. Визуализация через pandas:")

# Гистограмма через pandas
plt.figure(figsize=(10, 6))
df['radius_mean'].hist()
plt.title('Гистограмма распределения radius_mean')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.savefig('histogram.png')  # Сохраняем в файл
plt.close()

# Диаграмма рассеивания через pandas
plt.figure(figsize=(10, 6))
df.plot(kind='scatter', x='radius_mean', y='texture_mean')
plt.title('Диаграмма рассеивания radius_mean и texture_mean')
plt.xlabel('Средний радиус')
plt.ylabel('Средняя текстура')
plt.savefig('scatter.png')  # Сохраняем в файл
plt.close()

# Диаграмма "ящик с усиками" через pandas
plt.figure(figsize=(10, 6))
df.boxplot(column='radius_mean', by='diagnosis')
plt.title('Диаграмма "ящик с усиками" для radius_mean по diagnosis')
plt.suptitle('')  # Убираем автоматический заголовок
plt.savefig('boxplot.png')  # Сохраняем в файл
plt.close()

# 12. Pandas-Profiling отчет
print("\nЗадание 12. Создание отчета Pandas-Profiling")
# Выбираем основные признаки для отчета
df_report = df[[
    'diagnosis',  # Целевая переменная
    # Основные характеристики
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    # Важные дополнительные характеристики
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    # Худшие значения
    'radius_worst', 'texture_worst', 'perimeter_worst'
]].rename(columns={
    'diagnosis': 'Diagnosis (M/B)',
    'radius_mean': 'Mean Radius',
    'texture_mean': 'Mean Texture',
    'perimeter_mean': 'Mean Perimeter',
    'area_mean': 'Mean Area',
    'smoothness_mean': 'Mean Smoothness',
    'compactness_mean': 'Mean Compactness',
    'concavity_mean': 'Mean Concavity',
    'radius_worst': 'Worst Radius',
    'texture_worst': 'Worst Texture',
    'perimeter_worst': 'Worst Perimeter'
})

profile = ProfileReport(
    df_report, 
    title="Breast Cancer Wisconsin (Diagnostic) Dataset Analysis",
    minimal=False,  # Полный отчет вместо минимального
    explorative=True  # Включить все исследовательские графики
)
profile.to_file("breast_cancer_report.html")

# 13. AutoViz графики
print("\nЗадание 13. Создание графиков AutoViz")

# Создаем директорию для графиков, если её нет
if not os.path.exists('autoviz_plots'):
    os.makedirs('autoviz_plots')

AV = AutoViz_Class()
dft = AV.AutoViz(
    filename="",
    sep=",",
    depVar="diagnosis",  # Целевая переменная
    dfte=df,  # Наш датафрейм
    header=0,
    verbose=2,  # Увеличиваем уровень подробности вывода
    lowess=False,
    chart_format="png",  # Формат графиков
    save_plot_dir=os.path.abspath("autoviz_plots"),  # Абсолютный путь к директории
    max_rows_analyzed=569,  # Анализируем все строки
    max_cols_analyzed=32  # Анализируем все колонки
)

"""
Выводы:
1. Набор данных содержит 569 записей и 32 признака
2. Отсутствуют пропущенные значения в исходном наборе
3. Категориальным признаком является только 'diagnosis'
4. Созданы три набора с разными признаками для анализа
5. Выполнены все типы объединений данных
6. Проведены различные операции группировки
7. Добавлены новые информативные признаки
8. Создан составной индекс
9. Выполнено кодирование категориального признака
10. Построены все необходимые визуализации
"""
