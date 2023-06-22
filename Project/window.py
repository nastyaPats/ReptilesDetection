from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk
from reptiles_detection import *


class Gui:
    def __init__(self, master):
        self.master = master
        self.create_widgets()
        self.filepath = ''

    def create_widgets(self):
        self.select = Button(self.master, text="select an image", command=self.select_image)
        self.select.pack()
        self.detect = Button(self.master, text="detect", command=self.detect_image)
        self.detect.pack()
        self.canvas = Canvas(self.master, width= 400, height=400, bg="grey")
        self.canvas.pack()

    def select_image(self):
        self.file_path = filedialog.askopenfilename()
        des = Image.open(self.file_path)
        bg_image = ImageTk.PhotoImage(des)
        self.canvas.bg_image = bg_image
        self.canvas.create_image(200, 200, image=self.canvas.bg_image)

    def detect_image(self):
        self.p = Predict()
        self.p.detect(self.file_path)


if __name__ == "__main__":
    root = Tk()
    root.winfo_toplevel().title('Detection')
    my_gui = Gui(root)
    root.mainloop()





