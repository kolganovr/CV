# Фильтр для наложения изображения на лицо в видеопотоке с веб-камеры

## Авторы
Колганов РА

Когтев ВД

Егоров ТМ

Группа: М3О-314Б-21
## Задача
Создать скрипт на Python с использованием библиотеки OpenCV, который будет выполнять следующее:

- Получать видеопоток с веб-камеры
- Обрабатывать каждый кадр видео 
  - Находить лица на кадре с помощью хара-каскадов
  - Накладывать заранее подготовленное изображение поверх найденных лиц
- Выводить результат (кадр с наложенным изображением) в отдельное окно

## Решение

1. Импортируем необходимые библиотеки OpenCV (`cv2`) и NumPy (`np`).

2. Загружаем изображение, которое будем накладывать на лицо:
```python
overlay_img = cv2.imread('kitti_face-transformed.png', cv2.IMREAD_UNCHANGED) 
```
Флаг `cv2.IMREAD_UNCHANGED` позволяет сохранить альфа канал изображения.

3. Инициализируем веб-камеру:
```python 
cap = cv2.VideoCapture(0)
```

4. Запускаем бесконечный цикл для обработки кадров:

5. Считываем очередной кадр:
```python
ret, frame = cap.read()
```

6. Обнаруживаем лица на кадре:
```python
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(frame, 1.3, 5)
```
Используем готовый хара-каскад для распознавания лиц.

7. Циклом обрабатываем каждое найденное лицо:

8. Корректируем размер и положение прямоугольника лица:
```python  
difX = int(w/2)
difY = int(h/2)
x = x - int(difX/2)  
y = y - difY
w = w + difX
h = h + difY
```
Это нужно для более точного наложения изображения.

9.  Масштабируем overlay под размер найденного лица:
```python
overlay_resized = cv2.resize(overlay_img, (w, h))
```

10.  Создаем маску из альфа канала: 
```python
overlay_mask = overlay_resized[:,:,3] 
```

11.  Проверяем, что лицо не выходит за границы кадра.

12.  Накладываем изображение на кадр:
```python
bg_part = cv2.bitwise_and(frame[y:y+h, x:x+w], frame[y:y+h, x:x+w], mask=255-overlay_mask)
fg_part = cv2.bitwise_and(overlay_resized[:,:,:3], overlay_resized[:,:,:3], mask=overlay_mask)
frame[y:y+h, x:x+w] = bg_part + fg_part
```
Используем битовые операции AND для смешивания foreground и background частей.

13.  Выводим результат в отдельное окно:
```python
cv2.imshow('face_detect', frame)
```

14.  Выходим из цикла по нажатию ESC. 

15.  Освобождаем ресурсы:
```python
cap.release()
cv2.destroyAllWindows() 
```

## Результат
В результате работы скрипта мы получаем окно с видеопотоком, в котором на лица накладывается заданное изображение.