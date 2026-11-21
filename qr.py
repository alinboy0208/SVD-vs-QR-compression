import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


def qr_decomposition(A):
    m, n = A.shape
    q_vectors = []
    for i in range(n):
        v = A[:, i].copy() 
        for q in q_vectors:
            proj = (q @ v) * q
            v = v - proj
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            q_i = v / norm
        else:
            q_i = np.zeros_like(v)
        q_vectors.append(q_i)
    Q = np.column_stack(q_vectors)
    R = Q.T @ A
    return Q, R

img = Image.open("jpan.jpg").convert("RGB").resize((250, 250))
matrix = np.array(img, dtype=float)
qr_data = [qr_decomposition(matrix[:, :, c]) for c in range(3)]

def show_image(event=None):
    val = float(entry_k.get())
    max_k = qr_data[0][0].shape[1]
    k = max(1, int(round(val * max_k))) if 0 < val < 1 else max(1, min(int(round(val)), max_k))

    channels = [Q[:, :k] @ R[:k, :] for Q, R in qr_data]
    A_color = np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

    photo = ImageTk.PhotoImage(Image.fromarray(A_color))
    lbl_img.config(image=photo)
    lbl_img.image = photo


root = tk.Tk()
root.title("QR Decomposition")

tk.Label(root, text="Số chiều đặc trưng giữ lại (Latent Features):").pack()
entry_k = tk.Entry(root)
entry_k.insert(0, "20")
entry_k.pack()

tk.Button(root, text="Xem ảnh", command=show_image).pack(pady=5)

lbl_img = tk.Label(root)
lbl_img.pack()

root.bind("<Return>", show_image)
root.bind("<KP_Enter>", show_image)
entry_k.focus_force()
entry_k.select_range(0, tk.END)

show_image()
root.mainloop()