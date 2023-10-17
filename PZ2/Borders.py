import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

# Использовать ли веб-камеру
USE_WEB_CAM = False

# Путь к изображению
PATH = "PZ2/cat-portrait.jpg"

def histograms(img):
    # Получаем гистограмму по каждому из цветов RGB
    hist_r = np.histogram(img[..., 0], bins=256)[0]
    hist_g = np.histogram(img[..., 1], bins=256)[0]
    hist_b = np.histogram(img[..., 2], bins=256)[0]

    # Выводим 3 гистограммы на одном графике
    plt.subplot(141, title='Histograms')
    plt.plot(hist_r, color='red')
    plt.plot(hist_g, color='green')
    plt.plot(hist_b, color='blue')

    # Выводим изображение
    plt.subplot(142, title='Image RGB')
    plt.imshow(img)

    # Переводим изображение в HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Выводим изображение HSV
    plt.subplot(143, title='Image HSV')
    plt.imshow(img_hsv)

    # Получаем гистограмму по каждому из цветов HSV
    hist_h = np.histogram(img_hsv[..., 0], bins=256)[0]

    # Выводим гистограмму HSV
    plt.subplot(144, title='HSV histogram')
    plt.plot(hist_h, color='black')

    # Выводим график
    plt.show()

def clip(img):
    # Определяем границы опытным путем
    maxBorder = 255
    minBorder = 165

    # Делаем все пиксели, которые не попали в границы черными
    img[(img[..., 0] < minBorder) | (img[..., 0] > maxBorder)] = 0

    # Выводим результат
    plt.plot(title='Result')
    plt.imshow(img)

    # Выводим график
    plt.show()

    # Возвращаем результат
    return img

def gausian_canny(img):
    # Применяем фильтр Гаусса
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # Обнаружение границ с помощью фильтра Кэнни
    img_canny = cv2.Canny(img_gaussian, 100, 200)

    # Возвращаем результат
    return img_canny

# Выделение границ на изображении
def borders(img, size=301):
    # Применяем фильтр Гаусса
    blurred = cv2.GaussianBlur(img, (size, size), 0)

    # Выделяем границы
    edges_highlighted = cv2.subtract(img, blurred)

    # Возвращаем результат
    return edges_highlighted

def compare(img1, img2, name1='Image1', name2='Image2'):
    # Выводим изображение 1
    plt.subplot(121, title=name1)
    plt.imshow(img1)

    # Выводим изображение 2
    plt.subplot(122, title=name2)
    plt.imshow(img2)

    # Выводим график
    plt.show()

if __name__ == "__main__":
    if not USE_WEB_CAM:
        # Загрузка изображения
        img = io.imread(PATH)

        # Получение гистограмм
        histograms(img)

        # Обрезка изображения
        clipped_img = clip(img)

        # Применение фильтра Гаусса и обнаружение границ
        canny = gausian_canny(img)

        # Выделение границ на изображении
        edges = borders(img, 301)

        # Сравнение изображений
        compare(canny, edges, 'Canny', 'Borders')
    else:
        # Инициализация веб-камеры
        cap = cv2.VideoCapture(0)
        while True:
            # Получение кадра с веб-камеры
            ret, frame = cap.read()

            # Выделение границ на изображении
            bordersImg = borders(frame, 37)

            # Отображение результата
            cv2.imshow('frame', bordersImg)

            # Выход из программы по нажатию ESC
            if cv2.waitKey(1) == 27:
                break

        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()