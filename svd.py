import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


def svd_decomposition(A, At):
    At_A = At @ A
    eigenvalues, eigenvectors = np.linalg.eigh(At_A)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    singular_values = []
    right_vectors = []
    left_vectors = []

    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 1e-8:
            sigma = np.sqrt(eigenvalues[i])
            singular_values.append(sigma)

            vec = eigenvectors[:, i]
            v_norm = vec / np.linalg.norm(vec)
            right_vectors.append(v_norm)

            u = (1.0 / sigma) * (A @ v_norm)
            left_vectors.append(u)

    Sigma = np.diag(singular_values)
    U = np.column_stack(left_vectors)
    Vt = np.column_stack(right_vectors).T
    return U, Sigma, Vt


img = Image.open("jpan.jpg").convert("RGB").resize((250, 250))
matrix = np.array(img, dtype=float)
svd_data = [svd_decomposition(matrix[:, :, c], matrix[:, :, c].T) for c in range(3)] #Lưu ý nén sang RGB không gian 3D


def show_image(event=None):
    k = int(entry_k.get())
    channels = [U[:, :k] @ S[:k, :k] @ Vt[:k, :] for U, S, Vt in svd_data]
    A_color = np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

    photo = ImageTk.PhotoImage(Image.fromarray(A_color))
    lbl_img.config(image=photo)
    lbl_img.image = photo


root = tk.Tk()
root.title("Singular Value Decomposition")
tk.Label(root, text="Độ nén thông tin (Top Features):").pack()
entry_k = tk.Entry(root)
entry_k.insert(0, "20")
entry_k.pack()

entry_k.bind("<Return>", show_image)
entry_k.bind("<KP_Enter>", show_image)
entry_k.select_range(0, tk.END)

tk.Button(root, text="Xem ảnh", command=show_image).pack()

lbl_img = tk.Label(root)
lbl_img.pack()

show_image()
root.mainloop()
