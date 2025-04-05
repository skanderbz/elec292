import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pickle
import os

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
                import numpy as np
                from scipy.stats import skew, kurtosis

                # Load CSV
                df = pd.read_csv(self.file_path)

                # Rename expected columns if needed
                rename_map = {
                    'Time (s)': 'time',
                    'Linear Acceleration x (m/s^2)': 'x',
                    'Linear Acceleration y (m/s^2)': 'y',
                    'Linear Acceleration z (m/s^2)': 'z'
                }
                df.rename(columns=rename_map, inplace=True)

                df = df.dropna()
                df[["x", "y", "z"]] = df[["x", "y", "z"]].rolling(window=5, center=True).mean()
                df.dropna(inplace=True)

                # Segment into 5-second windows (assuming 100Hz)
                data = df[["x", "y", "z"]].values
                sampling_rate = 100
                window_size = 5 * sampling_rate

                segments = []
                for i in range(0, len(data) - window_size, window_size):
                    segments.append(data[i:i + window_size])

                # Feature extraction
                def extract_features(window):
                    features = []
                    for axis in range(3):
                        signal = window[:, axis]
                        features.extend([
                            np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
                            np.median(signal), np.ptp(signal), np.var(signal),
                            skew(signal), kurtosis(signal), np.sum(signal ** 2)
                        ])
                    return features

                feature_matrix = np.array([extract_features(w) for w in segments])

                # Load trained model
                model_path = os.path.join(os.path.dirname(__file__), "..", "model", "Trained_Model.pkl")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                # Predict
                predictions = model.predict(feature_matrix)

                # Save predictions to CSV
                output_path = self.file_path.replace(".csv", "_predictions.csv")
                pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)

                messagebox.showinfo("Success", f"✅ Predictions saved to:\n{output_path}")
                self.predictions = predictions
                self.plot_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file: {e}")
        else:
            messagebox.showwarning("No File", "Please select a file first.")

    def plot_results(self):
        if hasattr(self, "predictions") and self.file_path:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np

            # Load the CSV
            df = pd.read_csv(self.file_path)

            # Rename expected columns
            rename_map = {
                'Time (s)': 'time',
                'Linear Acceleration x (m/s^2)': 'x',
                'Linear Acceleration y (m/s^2)': 'y',
                'Linear Acceleration z (m/s^2)': 'z'
            }
            df.rename(columns=rename_map, inplace=True)
            df.dropna(inplace=True)

            # Smooth the signals
            df[["x", "y", "z"]] = df[["x", "y", "z"]].rolling(window=5, center=True).mean()
            df.dropna(inplace=True)

            time = df["time"].values
            x = df["x"].values
            y = df["y"].values
            z = df["z"].values

            predictions = self.predictions
            sampling_rate = 100
            window_size = 5 * sampling_rate

            # Build prediction step line
            pred_time = []
            pred_vals = []

            for i, p in enumerate(predictions):
                val = 1 if str(p).lower() == "jumping" or p == 1 else 0
                start_idx = i * window_size
                if start_idx < len(time):
                    pred_time.append(time[start_idx])
                    pred_vals.append(val)

            if len(pred_time) > 0:
                final_idx = (len(predictions)) * window_size
                if final_idx < len(time):
                    pred_time.append(time[final_idx])
                    pred_vals.append(pred_vals[-1])


            # === Plot ===
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(time, x, label="x_acc", color='blue', linewidth=1)
            ax1.plot(time, y, label="y_acc", color='orange', linewidth=1)
            ax1.plot(time, z, label="z_acc", color='red', linewidth=1)
            ax1.set_ylabel("acceleration (m/s²)")
            ax1.set_xlabel("Time (s)")
            ax1.grid(True)

            # Secondary axis for classification
            ax2 = ax1.twinx()
            ax2.step(pred_time, pred_vals, where='post', linestyle='--', color='green', label="jumping / (not walking)", linewidth=2)
            ax2.set_ylabel("classification 1 = jumping, 0 = walking")
            ax2.set_ylim(-0.2, 1.2)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            plt.title("Acceleration Data for interval")
            plt.tight_layout()
            plt.show()

        else:
            messagebox.showwarning("No Predictions", "You need to process a file first.")
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()


