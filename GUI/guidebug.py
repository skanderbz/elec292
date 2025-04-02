import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Graph Viewer")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Upload CSV File to Display Graph", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Browse", command=self.upload_file, width=20)
        self.upload_button.pack(pady=10)

        self.canvas = None
        self.file_path = None
        self.df = None

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if self.file_path:
            try:
                # Read the CSV file
                self.df = pd.read_csv(self.file_path)

                # Print column names to verify
                print("Columns found:", self.df.columns)

                # Check if the necessary columns exist
                required_columns = [
                    'Time (s)',
                    'Linear Acceleration x (m/s^2)',
                    'Linear Acceleration y (m/s^2)',
                    'Linear Acceleration z (m/s^2)',
                    'Absolute acceleration (m/s^2)'
                ]
                
                if not all(col in self.df.columns for col in required_columns):
                    messagebox.showerror("Error", f"CSV file must contain the columns: {', '.join(required_columns)}")
                    return

                self.plot_graph()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read the file: {e}")
        else:
            messagebox.showwarning("No File", "No file was selected. Please try again.")

    def plot_graph(self):
        if self.df is None:
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plotting the graph
        ax.plot(self.df['Time (s)'], self.df['Linear Acceleration x (m/s^2)'], label='Linear Acc X', color='r')
        ax.plot(self.df['Time (s)'], self.df['Linear Acceleration y (m/s^2)'], label='Linear Acc Y', color='g')
        ax.plot(self.df['Time (s)'], self.df['Linear Acceleration z (m/s^2)'], label='Linear Acc Z', color='b')
        ax.set_title("CSV Data Plot")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.legend(loc='upper right')
        ax.grid(True)

        # Display the plot in the tkinter window
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
