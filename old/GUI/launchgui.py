import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Walking/Jumping Classifier")

        # Set up window size
        self.root.geometry("400x200")

        # Title Label
        self.label = tk.Label(root, text="Upload CSV File for Classification", font=("Helvetica", 14))
        self.label.pack(pady=10)

        # Browse Button
        self.upload_button = tk.Button(root, text="Browse", command=self.upload_file, width=20)
        self.upload_button.pack(pady=10)

        # Process Button (disabled for now)
        self.process_button = tk.Button(root, text="Process File", command=self.process_file, width=20, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        # Plot Button (disabled for now)
        self.plot_button = tk.Button(root, text="Plot Results", command=self.plot_results, width=20, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        self.file_path = None

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if self.file_path:
            messagebox.showinfo("File Selected", f"Selected file: {self.file_path}")
            self.process_button.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("No File", "No file was selected. Please try again.")

    def process_file(self):
        if self.file_path:
            try:
                # Read the CSV file
                df = pd.read_csv(self.file_path)
                
                # Display a success message
                messagebox.showinfo("Success", "File successfully loaded and ready for processing!")
                
                # Enable the Plot Button
                self.plot_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read the file: {e}")
        else:
            messagebox.showwarning("No File", "Please select a file first.")

    def plot_results(self):
        messagebox.showinfo("Plot", "Plotting feature is not implemented yet.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
