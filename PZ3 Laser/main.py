import cv2
import numpy as np

DEBUG = False # True - Настройка диапазонов маски с помощью ползунков, False - Использование диапазонов маски из maskParams

USE_WEB = True # True - Использовать фотографии из папки USE_WEB_PHOTOS, False - Использовать фотографии из папки imgs

maskParams = {
    'H_min': 139,
    'H_max': 148,
    'S_min': 0,
    'S_max': 29,
    'V_min': 255,
    'V_max': 255
}

# Загрузка изображений
imgs = []
if USE_WEB:
    for i in range(5):
        with np.load(f'PZ3 Laser/USE_WEB_PHOTOS/img_{i}.npz') as data:
            imgs.append(data['frame'])
else:
    for i in range(1, 9):
        with np.load(f'PZ3 Laser/imgs/img_{i}.npz') as data:
            imgs.append(data['frame'])

def createTrackbars():
    # Создаем окно с ползунками для настройки параметров маски
    cv2.namedWindow('mask')
    cv2.createTrackbar('H_min', 'mask', 0, 255, lambda x: x)
    cv2.createTrackbar('H_max', 'mask', 0, 255, lambda x: x)
    cv2.createTrackbar('S_min', 'mask', 0, 255, lambda x: x)
    cv2.createTrackbar('S_max', 'mask', 0, 255, lambda x: x)
    cv2.createTrackbar('V_min', 'mask', 0, 255, lambda x: x)
    cv2.createTrackbar('V_max', 'mask', 0, 255, lambda x: x)

    # Set default values for all trackbars
    cv2.setTrackbarPos('H_min', 'mask', 0)
    cv2.setTrackbarPos('H_max', 'mask', 255)
    cv2.setTrackbarPos('S_min', 'mask', 0)
    cv2.setTrackbarPos('S_max', 'mask', 255)
    cv2.setTrackbarPos('V_min', 'mask', 0)
    cv2.setTrackbarPos('V_max', 'mask', 255)

def getTrackbarValues():
    # Получаем значения с ползунков
    H_min = cv2.getTrackbarPos('H_min', 'mask')
    H_max = cv2.getTrackbarPos('H_max', 'mask')
    S_min = cv2.getTrackbarPos('S_min', 'mask')
    S_max = cv2.getTrackbarPos('S_max', 'mask')
    V_min = cv2.getTrackbarPos('V_min', 'mask')
    V_max = cv2.getTrackbarPos('V_max', 'mask')

    return H_min, S_min, V_min, H_max, S_max, V_max

def applyMask(img):
    # Получаем значения с ползунков
    H_min, S_min, V_min, H_max, S_max, V_max = getTrackbarValues()
    # Создаем маску по параметрам hsv
    mask = cv2.inRange(img, (H_min, S_min, V_min), (H_max, S_max, V_max))

    # Применяем маску к изображению
    img_result = cv2.bitwise_and(img, img, mask=mask)

    # Выводим изображения
    cv2.imshow('mask', mask)

    # Переводим в RGB
    img_result = cv2.cvtColor(img_result, cv2.COLOR_HSV2RGB)
    cv2.imshow('img_result', img_result)

    # Для выхода из цикла нажать клавишу esc
    if cv2.waitKey(1) == 27:
        return img_result


def getResult(img):
    # Переводим изображение в HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if not DEBUG: # debug false
        # Создаем маску по параметрам hsv
        mask = cv2.inRange(img_hsv, (maskParams['H_min'], maskParams['S_min'], maskParams['V_min']),
                           (maskParams['H_max'], maskParams['S_max'], maskParams['V_max']))
        
        # Применяем маску к изображению
        img_result = cv2.bitwise_and(img, img, mask=mask)
        return img_result

    # Создаем окно с ползунками для настройки параметров маски
    createTrackbars()

    img_result = None
    # Применяем маску к изображению
    while img_result is None:
        img_result = applyMask(img_hsv)

    # Закрываем окно с ползунками
    cv2.destroyAllWindows()

    return img_result

# Получаем центр лазера
def getCenter(img):
    img_result = getResult(img)

    # Получаем координаты лазера
    xCenter = 0
    yCenter = 0
    count = 0
    for i in range(img_result.shape[0]):
        for j in range(img_result.shape[1]):
            if img_result[i, j, 0] != 0 and img_result[i, j, 1] != 0 and img_result[i, j, 2] != 0:
                xCenter += i
                yCenter += j
                count += 1

    if count > 0:
        # Получаем средние координаты лазера
        xCenter = int(xCenter / count)
        yCenter = int(yCenter / count)
    else:
        return [-1, -1]

    return [xCenter, yCenter]

# Рисует маленький красный круг в центре лазера
def showResult(imgNum, center):
    img = imgs[imgNum]

    # Рисуем красный круг в центре лазера
    if center != [-1, -1]:
        cv2.circle(img, (center[1], center[0]), 5, (0, 0, 255), -1)
        
    cv2.imshow('img_result', img)

    # Сохраняем результат как картинку jpg в папку Results
    cv2.imwrite(f'PZ3 Laser/Results/result_{imgNum}.jpg', img)

    # Ждем нажатия любой клавиши для закрытия окна
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    centers = []
    for imNum in range(len(imgs)):
        print(f'Image index {imNum}', end='\t')

        # Получаем центр лазера
        center = getCenter(imgs[imNum])
        centers.append(center)

        # Выводим результат и сохраняем в файл
        print(f'X: {center[1]}, Y: {center[0]}')
        showResult(imNum, center)

    # Считаем СКО
    sum = 0
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            sum += np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
    print(f'СКО: {sum / (len(centers) * (len(centers) - 1) / 2)}')

    # Считаем стандартное отклонение
    std = np.std(centers, axis=0)
    print(f'Стандартное отклонение: {std}')

    # Радиус разброса
    maxDist = 0
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            if dist > maxDist:
                maxDist = dist
    
    print(f'Радиус разброса: {maxDist/2}')