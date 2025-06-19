import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime

import tenseal as ts
import numpy as np
from extract_features import extract_feature
from encrypted_classifier import classify_encrypted

LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

def classify_image_encrypted(img_path):
    features = extract_feature(img_path)
    context = create_context()
    enc_vector = ts.ckks_vector(context, features)
    prediction_vector = classify_encrypted(enc_vector)

    detected = []
    log_lines = []
    for i, score in enumerate(prediction_vector):
        line = f"{LABELS[i]} → {score:.3f}"
        log_lines.append(line)
        if score > 0:
            detected.append(LABELS[i])

    summary = " | ".join(detected) if detected else " No diseases detected"
    return summary, log_lines

class EyeDiseaseApp:
    def __init__(self, root):  #  تعديل هنا
        self.root = root
        self.root.title("Eye Disease Classifier (CKKS Encrypted)")
        self.root.geometry("750x650")
        self.image_path = None

        tk.Label(root, text="🧬 Encrypted Eye Disease Classifier", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Label(root, text="Patient Name:", font=("Arial", 12)).pack()
        self.patient_name_var = tk.StringVar()
        self.entry_name = tk.Entry(root, textvariable=self.patient_name_var, font=("Arial", 12), width=40)
        self.entry_name.pack(pady=5)

        self.btn_choose = tk.Button(root, text=" Choose Fundus Image", command=self.load_image)
        self.btn_choose.pack(pady=5)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        self.btn_predict = tk.Button(root, text=" Encrypted Diagnosis", command=self.predict)
        self.btn_predict.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.result_label.pack(pady=10)

        self.btn_save = tk.Button(root, text= Save Report", command=self.save_report, state=tk.DISABLED)
        self.btn_save.pack(pady=5)

        self.report_log = ""

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            self.canvas.configure(image=img)
            self.canvas.image = img
            self.result_label.config(text="")
            self.report_log = ""
            self.btn_save.config(state=tk.DISABLED)

    def predict(self):
        if not self.image_path:
            messagebox.showerror("Missing Image", "Please select an image.")
            return

        patient_name = self.patient_name_var.get().strip()
        if not patient_name:
            messagebox.showwarning("Missing Name", "Please enter the patient's name.")
            return

        try:
            summary, lines = classify_image_encrypted(self.image_path)
            self.result_label.config(text=f"Diagnosis: {summary}")
            log = f" Patient Name: {patient_name}\n"
            log += f" Image: {os.path.basename(self.image_path)}\n"
            log += f" Date: {datetime.datetime.now()}\n\n"
            log += "\n".join(lines)
            log += f"\n\n Final Result: {summary}\n"
            self.report_log = log
            self.btn_save.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def save_report(self):
        if not self.report_log:
            return
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        patient = self.patient_name_var.get().strip().replace(" ", "_")
        filename = f"report_{patient}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(os.path.join(report_dir, filename), "w", encoding="utf-8") as f:
            f.write(self.report_log)
        messagebox.showinfo("Saved", f"Report saved as: {filename}")

#  تشغيل التطبيق
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeDiseaseApp(root)
    root.mainloop()
