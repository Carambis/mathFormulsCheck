import os

# Force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Debug messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import symbol as sym


def symbols_extract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, bmp_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(bmp_image, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    symbols = list()

    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            symbol_crop = bmp_image[y:y + h, x:x + w]
            symbols.append(sym.Symbol(x, y, x + w, y + h, symbol_crop))

    return symbols


def check_symbols_inside(symbols):
    size = len(symbols)
    for i in range(0, size):
        for j in range(i + 1, size):
            symbol = symbols[i]
            symbol_next = symbols[j]
            if symbol_next.get_lx() >= symbol.get_lx() and symbol_next.get_ly() >= symbol.get_ly() and symbol_next.get_hx() <= symbol.get_hx() and symbol_next.get_hy() <= symbol.get_hy():
                replace_symbol_inside(symbol, symbol_next)
            elif symbol_next.get_lx() <= symbol.get_lx() and symbol_next.get_ly() <= symbol.get_ly() and symbol_next.get_hx() >= symbol.get_hx() and symbol_next.get_hy() >= symbol.get_hy():
                replace_symbol_inside(symbol_next, symbol)


def replace_symbol_inside(symbol, symbol_next):
    image = symbol.get_img().copy()
    mask = np.full((symbol_next.get_hy() - symbol_next.get_ly(), symbol_next.get_hx() - symbol_next.get_lx()), 255)
    start_x = symbol_next.get_lx() - symbol.get_lx()
    start_y = symbol_next.get_ly() - symbol.get_ly()
    image[start_y:(start_y + mask.shape[0]), start_x:(start_x + mask.shape[1])] = mask
    symbol.set_img(image)


def resize_images(source_image, symbols, output_size):
    for symbol in symbols:
        img = symbol.get_img().copy()
        symbol.set_img(cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA))
        lx = symbol.get_lx()
        ly = symbol.get_ly()
        hx = symbol.get_hx()
        hy = symbol.get_hy()
        w = hx - lx
        h = hy - ly

        cv2.rectangle(source_image, (lx, ly), (hx, hy), (70, 0, 0), 1)
        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if w > h:
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = img
        elif w < h:
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = img
        else:
            letter_square = letter_crop

        symbol.set_img(cv2.resize(letter_square, (output_size, output_size), interpolation=cv2.INTER_AREA))


def show_image(symbols):
    size = len(symbols)
    for i in range(0, size):
        cv2.imshow(str(i), symbols[i].get_img())

    cv2.waitKey(0)


if __name__ == "__main__":
    img = cv2.imread("test1.png")

    symbols = symbols_extract(img)
    check_symbols_inside(symbols)
    symbols.sort(key=lambda symbol: symbol.get_lx(), reverse=False)

    out_size = 28

    resize_images(img, symbols, out_size)
    show_image(symbols)
