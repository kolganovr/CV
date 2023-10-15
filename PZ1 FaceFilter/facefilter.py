import cv2

IMAGE = 'PZ1 FaceFilter/kitty_face-transformed.png'

# Загрузка изображения, которое будет накладываться 
overlay_img = cv2.imread(IMAGE, cv2.IMREAD_UNCHANGED)

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

# Цикл обработки кадров запускается, если изображение для наложения загружено и веб-камера работает
while overlay_img is not None and cap.isOpened():
    # Считывание кадра
    ret, frame = cap.read()

    # Преобразование цветового пространства кадра в RGBA для корректного отображения прозрачности
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = frame

    # Обнаружение лиц на кадре с помощью хара-каскадов
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(img, 1.3, 5)

    # Накладывание изображения на лицо
    for (x, y, w, h) in faces:
        # Изменение размера лица для более точного накладывания изображения
        difX = int(w/2)
        difY = int(h/2)
        x = x - int(difX/2)
        y = y - difY
        w = w + difX
        h = h + difY

        # Масштабирование overlay под размер лица
        overlay_resized = cv2.resize(overlay_img, (w, h))
        
        # Создание маски из альфа канала
        overlay_mask = overlay_resized[:,:,3]

        # Если лицо выходит за границы кадра, то оно не будет обработано
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            continue
        
        # Наложение изображения на кадр
        bg_part = cv2.bitwise_and(frame[y:y+h, x:x+w], frame[y:y+h, x:x+w], mask=255-overlay_mask)
        fg_part = cv2.bitwise_and(overlay_resized[:,:,:3], overlay_resized[:,:,:3], mask=overlay_mask)
        frame[y:y+h, x:x+w] = bg_part + fg_part

    # Отображение результата
    cv2.imshow('face_detect', frame)

    # Выход из программы по нажатию ESC
    if cv2.waitKey(1) == 27:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
