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
from pandas_profiling import ProfileReport
from autoviz.AutoViz_Class import AutoViz_Class
# Проверка наличия файла
if not os.path.exists('breast_cancer.csv'):
    print("Ошибка: Файл breast_cancer.csv не найден в текущей директории")
    exit()
# 1. Загрузка набора данных
df = pd.read_csv('breast_cancer.csv')

# 2. Получение информации о н��боре данных
print("Информация о наборе данных:")
print(df.info())
print("\nПервые 5 строк данных:")
print(df.head())

# 3. Обработка пустых значений и дубликатов
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
# (выбраны наиболее значимые признаки для диагностики на основе медицинских показателей)
diagnostic_columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
df_diagnostic = df[diagnostic_columns]
print("\nНабор 3 - важные диагностические признаки:", df_diagnostic.columns.tolist())

# 5. Генерация нового набора и все виды объединений
df_new = df.sample(n=100).copy()  # берем случайные 100 записей
df_new['new_feature'] = range(len(df_new))  # добавляем новый признак

# Все виды объединений
merged_inner = pd.merge(df, df_new, on='id', how='inner')
merged_left = pd.merge(df, df_new, on='id', how='left')
merged_right = pd.merge(df, df_new, on='id', how='right')
merged_outer = pd.merge(df, df_new, on='id', how='outer')

# 6. Группировка и агрегация
# Группировка 1: Среднее по диагнозу
g1 = df.groupby('diagnosis').mean()

# Группировка 2: Несколько агрегирующих функций
g2 = df.groupby('diagnosis').agg({
    'radius_mean': ['min', 'max', 'mean', 'std'],
    'texture_mean': ['min', 'max', 'mean', 'std']
})

# Группировка 3: Подсчет количества
g3 = df.groupby('diagnosis').size()

# Группировка 4: Медиана
g4 = df.groupby('diagnosis').median()

# Группировка 5: Квантили
g5 = df.groupby('diagnosis').quantile([0.25, 0.75])

# 7. Новые признаки
df_features = df.copy()
# Площадь к периметру
df_features['area_perimeter_ratio'] = df_features['area_mean'] / df_features['perimeter_mean']
# Комплексный показатель текстуры
df_features['texture_severity'] = df_features['texture_mean'] * df_features['smoothness_mean']
# Индекс компактности
df_features['compactness_index'] = df_features['perimeter_mean']**2 / df_features['area_mean']

# 8. Составной индекс
df_indexed = df_features.copy()
df_indexed.set_index(['diagnosis', 'radius_mean'], inplace=True)

# 9. Кодирование категориальных признаков
# Способ 1: Label Encoding через map
df_encoded = df.copy()
df_encoded['diagnosis_coded'] = df_encoded['diagnosis'].map({'M': 1, 'B': 0})

# Способ 2: One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['diagnosis'])

# 10. Статистические данные
print("\nСтатистические данные:")
print(df.describe())

# 11. Визуализация через pandas
# Гистограмма
df['radius_mean'].plot(kind='hist', figsize=(10, 6))

# Диаграмма рассеивания
df.plot.scatter(x='radius_mean', y='texture_mean', figsize=(10, 6))

# Диаграмма "ящик с усиками"
df.boxplot(by='diagnosis', column='radius_mean', figsize=(10, 6))

# 12. Pandas-Profiling отчет
profile = ProfileReport(df, title="Breast Cancer Dataset Analysis")
profile.to_file("breast_cancer_report.html")

# 13. AutoViz графики
AV = AutoViz_Class()
dft = AV.AutoViz(
    filename="",
    sep=",",
    depVar="diagnosis",
    dfte=df,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="png"
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
