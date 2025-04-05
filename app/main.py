import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import appfunctions as af
import matplotlib.pyplot as plt
import numpy as np
import pickle

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Walking/Jumping Classifier")
        self.root.geometry("400x200")
        self.label = tk.Label(root, text="Upload CSV File for Classification", font=("Helvetica", 14))
        self.label.pack(pady=10)
        self.upload_button = tk.Button(root, text="Browse", command=self.upload_file, width=20)
        self.upload_button.pack(pady=10)
        self.process_button = tk.Button(root, text="Process File", command=self.process_file, width=20, state=tk.DISABLED)
        self.process_button.pack(pady=10)
        self.plot_button = tk.Button(root, text="Plot Results", command=self.plot_results, width=20, state=tk.DISABLED)
        self.plot_button.pack(pady=10)
        self.file_path = None
        self.predictions = None
        self.raw_df = None  
        self.processed_df = None

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
                df = af.csvtodf(self.file_path)
                self.raw_df = df.copy() 
                #EDIT SMOOTHING
                df_preprocessed = af.preprocess_df(df, rolling_window_size=100)
                self.processed_df = df_preprocessed.copy()

                #EDIT SPLIT
                features_df = af.split_and_extract_features(df_preprocessed, window_size=500, normalize_features=True)
                # print(features_df)

                model_path = os.path.join(os.path.dirname(__file__), "..", "model", "Trained_Model.pkl")
                predictions = af.predict_activity(features_df, model_path=model_path)
                self.predictions = predictions

                self.plot_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file: {e}")
        else:
            messagebox.showwarning("No File", "Please select a file first.")

    def plot_results(self):
        if self.predictions is not None and self.raw_df is not None:
            df = self.raw_df.copy().sort_values("time").reset_index(drop=True)
            time = df["time"].values
            x = df["x"].values
            y = df["y"].values
            z = df["z"].values

            sampling_rate = 500
            window_size = 1 * sampling_rate


            pred_time = []
            pred_vals = []
            for i, pred in enumerate(self.predictions):
                start_idx = i * window_size
                if start_idx < len(time):
                    pred_time.append(time[start_idx])
                    pred_vals.append(int(pred))
            if len(pred_time) > 0:
                final_idx = len(self.predictions) * window_size
                if final_idx < len(time):
                    pred_time.append(time[final_idx])
                    pred_vals.append(pred_vals[-1])

            # Plot .
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(time, x, label="x_acc", color="blue", linewidth=1)
            ax1.plot(time, y, label="y_acc", color="orange", linewidth=1)
            ax1.plot(time, z, label="z_acc", color="red", linewidth=1)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Acceleration (m/s²)")
            ax1.grid(True)

            # Plot 2
            ax2 = ax1.twinx()
            ax2.step(pred_time, pred_vals, where="post", linestyle="--", color="green", linewidth=2, label="Prediction")
            ax2.set_ylabel("Prediction (0 = walking, 1 = jumping)")
            ax2.set_ylim(-0.2, 1.2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            plt.title("Acceleration Data and Predicted Activity")
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("You need to process a file first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
