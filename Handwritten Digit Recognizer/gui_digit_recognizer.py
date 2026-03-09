from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np

# Load trained model
model = load_model('mnist.h5')


def preprocess(img):
    # convert to grayscale
    img = img.convert('L')
    img = np.array(img)

    # invert colors (MNIST expects white digit)
    img = 255 - img

    # remove weak pixels
    img[img < 50] = 0

    # find bounding box
    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[y_min:y_max, x_min:x_max]

    # resize digit to 20x20
    img = Image.fromarray(img)
    img = img.resize((20,20))

    img = np.array(img)

    # pad to 28x28
    img = np.pad(img, ((4,4),(4,4)), 'constant')

    # normalize
    img = img.reshape(1,28,28,1)
    img = img.astype('float32') / 255.0

    return img


def predict_digit(img):

    img = preprocess(img)

    if img is None:
        return 0,0

    res = model.predict(img, verbose=0)

    return np.argmax(res), np.max(res)


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Handwritten Digit Recognizer")

        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, pady=2)

        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 48))
        self.label.grid(row=0, column=1, padx=20)

        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.classify_btn.grid(row=1, column=1)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.button_clear.grid(row=1, column=0)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit")

    def classify_handwriting(self):

        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()

        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x,y,x1,y1))

        digit, acc = predict_digit(img)

        self.label.configure(text=str(digit) + ", " + str(int(acc*100)) + "%")

    def draw_lines(self, event):

        r = 8

        self.canvas.create_oval(
            event.x-r,
            event.y-r,
            event.x+r,
            event.y+r,
            fill='black'
        )


app = App()
app.mainloop()