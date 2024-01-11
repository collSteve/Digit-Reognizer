import tensorflow as tf
import tkinter as tk
import numpy as np

root = tk.Tk()

# Create a canvas widget
canvas = tk.Canvas(root, width=600, height=600, bg="black")
canvas.pack()

num_rows = 28
num_columns = 28

digit = np.zeros((num_rows, num_columns))

canvas_width = int(canvas.cget('width'))
canvas_height = int(canvas.cget('height'))

# Load model
model = tf.keras.models.load_model('trained/dr1.h5')

# Function to draw a grid on the canvas
def draw_grid(num_rows, num_columns):
    # Determine the row width and column width
    row_width = canvas_width / num_columns
    column_height = canvas_height / num_rows

    # Draw the horizontal grid lines
    for i in range(num_rows):
        canvas.create_line([(0, i * column_height), (canvas_width, i * column_height)], tag='grid_line', fill="grey")
    
    # Draw the vertical grid lines
    for i in range(num_columns):
        canvas.create_line([(i * row_width, 0), (i * row_width, canvas_height)], tag='grid_line', fill="grey")

def drag(event, color, tobe):
    x = event.x + event.widget.winfo_x()
    y = event.y + event.widget.winfo_y()

    row_width = canvas_width / num_columns
    column_height = canvas_height / num_rows
    
    x = int(x / row_width) * row_width
    y = int(y / column_height) * column_height

    digit[int(y / column_height)][int(x/ row_width)] = tobe

    canvas.create_rectangle(x, y, x + row_width, y + column_height, fill=color, tags='rectangle')


text_box = tk.Entry(root)
text_box.insert(0, f"Number: {-1}")
text_box.config(state='disabled')
text_box.pack()

def clear_grid():
    global digit
    digit = np.zeros((num_rows, num_columns))
    canvas.delete('rectangle')

def predict_digit():
    topred = np.array([digit]) 
    topred = topred[..., tf.newaxis]


    np.argmax(model.predict(topred, verbose=False))

    text_box.config(state='normal')
    text_box.delete(0, tk.END)
    text_box.insert(0, f"Number: {np.argmax(model.predict(topred))}")
    text_box.config(state='disabled')


# Create a button with an onclick event
button = tk.Button(root, text="Click Me", command=predict_digit)
# Place the button
button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_grid)
clear_button.pack()

# Call the function to draw the grid
draw_grid(num_rows, num_columns)

canvas.bind("<B1-Motion>", lambda e: drag(e, "red", 1))
canvas.bind("<B2-Motion>", lambda e: drag(e, "black", 0))

predict_digit()
# Start the tkinter event loop
root.mainloop()

print("Start")
