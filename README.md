# opencv_labs
## Задача 1. - done
Написать программу для определения
количества точек в изображении с яркостью выше
заданного значения (T). Исходное изображение
цветное и загружается с диска

## задача 2. - done
Реализовать «вручную» цветовое выделение
значимых частей на изображении: глаз, ушей, носа кота
– с помощью примитивов отрисовки (полигонов, кругов, 
эллипсов, линий).

## задача 3. - done
Реализовать адаптивное пороговое преобразование
для скана/фото текста неравномерной яркости

## задача 4. - done
Работа с гистограммой изображения. Визуализация
гистограммы изображения. Реализовать методы ручной коррекции
яркости и контраста, гамма коррекцию, а также автоматическую
коррекцию на основе метода эквализации гистограммы.

a) Ручная коррекция: alpha и beta задаются пользователем

b) Гамма коррекция. Результаты для gamma>1 и <1

c) Отрисовка гистограммы изображения с диска

e). Эквализация гистограммы: cv::equalizeHist добавить на предыдущей картинке

## задача 5. - done

Продемонстрировать возможности линейной фильтрации
изображений на основе: Гауссова фильтра, медианного, фильтра с
произвольным ядром, оператора Собеля, фильтра Лапласа. 
Продемонстрировать удаление равномерного шума и Гауссова.

• Гауссов(cv::blur) и медианный фильтр (cv::medianBlur).

• Фильтрация произвольной маской (cv::filter2D). 

• Выделение границ объектов (cv::Sobel).

• Фильтр Лапласа. Исследовать влияние входных параметров на
результат фильтрации, провести сравнение с фильтром Собеля.

• cv::randn – нормальное распределение, cv::randu – равномерное
распределение

## Задача 6. 
Продемонстрировать восстановление исходного изображения
по набору зашумленных изображений (уровень равномерного
аддитивного шума задает пользователь), определить, какое количество
изображений требуется для получения изображения приемлемого
качества.

## задача 7 - Done
1. Написать программу, выделяющую границы объектов
методом Кэнни.
Реализовать выделение контуров объектов по изображению границ,
полученному методом Кэнни для произвольного изображения, а также их
визуализацию поверх исходного изображения.

• Детектор границ Кэнни (cv::Canny) 

• Обнаружение и отрисовка контуров (cv:: findContours, cv::drawContours)

2. Размер объекта (маленький, средний, большой) относительно
общего размера изображения. - skip (просто посчитать отношение площади контура к площади изображения)
3. Форму объекта (квадрат, прямоугольник, треугольник, круг, овал). - idk, лень

## задача 8 -  DONE, линии
Реализовать поиск линий либо окружностей (на выбор) на
характерных изображениях, исследовать эффективность при различных
параметрах.

## задача 9  - ползунки - яркость контраст
Использовать детектор углов Харриса и детектор особых
точек Ши-Томаси, исследовать влияние преобразований изображений
(аффинные и перспективные преобразования, изменение яркости, 
контраста), а также параметров алгоритмов на результаты
обнаружения локальных особенностей.

## задача 10 - done
Реализовать поиск соответствия особых точек на двух
изображений (feature matching), исследовать работу в реальных
условиях.

## задача 11 - done
Реализовать сегментацию изображений с помощью алгоритма
watershed либо graphcut (на выбор), провести тестирование на
различных примерах при разных параметрах.

## задача 12
любая из 4-х на выбор
1. Реализовать алгоритм MNIST распознавания рукописных цифр, 
продемонстрировать несколько ошибочных примеров
2. Реализовать детектирование и распознавание номерных знаков
автомобилей по фотографии (алгоритм ALPR или любой другой)
3. Реализовать алгоритм распознавания лиц на основе метода
Виола-Джонса
4. Написать программу, определяющую возраст человека по его
фотографии (любой метод) -  Done

