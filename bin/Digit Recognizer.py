import tkinter as tk
from tkinter import messagebox
import subprocess

GRID_SIZE = 14
CELL_SIZE = 20
MAX_INTENSITY = 255
INTENSITY_STEP = 5
DRAW_INTERVAL_MS = 1

TEMP_INPUT_FILE = "temp_digit.txt"
RECOGNIZER_EXECUTABLE = "recognize_digit.exe"

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg='white')
        self.canvas.pack()

        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.update_position)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.is_drawing = False
        self.current_row_col = None

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_grid)
        clear_btn.pack(side=tk.LEFT, padx=5)

        recognize_btn = tk.Button(button_frame, text="Recognize", command=self.recognize_digit)
        recognize_btn.pack(side=tk.LEFT, padx=5)

    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.rects[i][j] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill='white', outline='lightgray')

    def start_drawing(self, event):
        self.is_drawing = True
        self.update_position(event)
        self.continuous_draw()

    def stop_drawing(self, event):
        self.is_drawing = False

    def update_position(self, event):
        row = event.y // CELL_SIZE
        col = event.x // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.current_row_col = (row, col)

    def continuous_draw(self):
        if self.is_drawing and self.current_row_col:
            row, col = self.current_row_col
            self.grid[row][col] = min(self.grid[row][col] + INTENSITY_STEP, MAX_INTENSITY)
            gray_value = 255 - self.grid[row][col]
            color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
            self.canvas.itemconfig(self.rects[row][col], fill=color)
            self.root.after(DRAW_INTERVAL_MS, self.continuous_draw)

    def clear_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.grid[i][j] = 0
                self.canvas.itemconfig(self.rects[i][j], fill='white')

    def save_temp_digit(self):
        flattened = [round(val / MAX_INTENSITY, 2) for row in self.grid for val in row]
        with open(TEMP_INPUT_FILE, 'w+') as f:
            f.write(f"{GRID_SIZE} {GRID_SIZE} {GRID_SIZE}\n")
            for i in range(GRID_SIZE):
                row = flattened[i * GRID_SIZE:(i + 1) * GRID_SIZE]
                f.write(' '.join(map(str, row)) + '\n')

    def recognize_digit(self):
        self.save_temp_digit()
        try:
            result = subprocess.run([RECOGNIZER_EXECUTABLE], capture_output=True, text=True, check=True)
            recognized = result.stdout.strip()
            messagebox.showinfo("Recognition Result", f"The digit is likely: {recognized}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to run recognizer:\n{e}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Could not find executable: {RECOGNIZER_EXECUTABLE}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
