class Symbol:
    def __init__(self, lx, ly, hx, hy, img):
        self.__lx = lx
        self.__ly = ly
        self.__hx = hx
        self.__hy = hy
        self.__img = img

    def set_img(self, img):
        self.__img = img

    def get_lx(self):
        return self.__lx

    def get_ly(self):
        return self.__ly

    def get_hx(self):
        return self.__hx

    def get_hy(self):
        return self.__hy

    def get_img(self):
        return self.__img


