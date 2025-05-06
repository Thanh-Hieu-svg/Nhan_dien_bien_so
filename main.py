import tkinter as tk
from tkinter import ttk
import mysql.connector
import subprocess
import os

# Kết nối CSDL
def connect_db():
    return mysql.connector.connect(
        host="localhost", 
        user="root", 
        password="", 
        database="bien_so_xe"
    )

def fetch_data(keyword=""):
    conn = connect_db()
    cursor = conn.cursor()
    query = "SELECT id, bien_so, thoi_gian FROM so_xedb"
    if keyword:
        query += " WHERE bien_so LIKE %s"
        cursor.execute(query, (f"%{keyword}%",))
    else:
        cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

def update_table():
    for row in tree.get_children():
        tree.delete(row)
    data = fetch_data(search_var.get())
    for row in data:
        tree.insert("", "end", values=row)

def search_plate():
    update_table()

def run_script(script_name):
    process = subprocess.Popen(["python", script_name])
    process.wait()
    update_table()

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Nhận diện biển số")
root.geometry("950x650")
root.configure(bg="#C8ECFE") 

# Custom style cho Treeview
style = ttk.Style()
style.theme_use("default")
style.configure("Custom.Treeview",
                background="white",
                fieldbackground="#C8ECFE",
                foreground="black",
                rowheight=30,
                font=("Arial", 12))
style.configure("Custom.Treeview.Heading",
                font=("Arial", 13, "bold"),
                background="#80B3D9",
                foreground="white")
style.map("Custom.Treeview",
          background=[("selected", "#1E90FF")])

# Frame chính
main_frame = tk.Frame(root, bg="#C8ECFE")
main_frame.pack(fill="both", expand=True)

# Tiêu đề
title_label = tk.Label(main_frame, 
                      text="NHẬN DIỆN BIỂN SỐ", 
                      font=("Arial", 20, "bold"), 
                      bg="#C8ECFE",
                      fg="#FF7F7F")
title_label.pack(pady=15)

# Frame tìm kiếm
search_frame = tk.Frame(main_frame, bg="#C8ECFE")
search_frame.pack(pady=10)

search_var = tk.StringVar()
search_entry = tk.Entry(search_frame, 
                       textvariable=search_var,
                       font=("Arial", 12),
                       width=30,
                       bd=1,
                       relief="solid",
                       bg="white")
search_entry.pack(side="left", ipady=4)


# Hàm hiệu ứng hover
def on_hover(event):
    event.widget.config(bg="#5A9EC9", fg="white")

def on_leave(event):
    event.widget.config(bg="#80B3D9", fg="white")

search_btn = tk.Button(
    search_frame, 
    text="Tìm kiếm", 
    command=search_plate,
    bg="#80B3D9",
    fg="white",
    activebackground="#5A9EC9",
    font=("Arial", 12),
    bd=0,
    relief="solid",
    height=1,      
    padx=10,
    pady=2,
    cursor="hand2"
)
search_btn.pack(side="left")

search_btn.bind("<Enter>", on_hover)
search_btn.bind("<Leave>", on_leave)

# Frame chứa nút chức năng
button_frame = tk.Frame(main_frame, bg="#C8ECFE")
button_frame.pack(pady=15)

buttons = {
    "Ảnh": "image.py",
    "Video": "video.py",
    "Camera": "camera.py"
}

for text, script in buttons.items():
    btn = tk.Button(
        button_frame, 
        text=text, 
        bg="#80B3D9",
        fg="white",
        activebackground="#C8ECFE",
        padx=20,
        pady=8,
        font=("Arial", 12, "bold"), 
        bd=3,
        relief="raised",
        width=12,
        cursor="hand2",
        command=lambda s=script: run_script(s)
    )
    btn.pack(side="left", padx=15)
    btn.bind("<Enter>", on_hover)
    btn.bind("<Leave>", on_leave)

# Frame chứa bảng dữ liệu
table_frame = tk.Frame(main_frame, bg="#C8ECFE")
table_frame.pack(pady=15, padx=20, fill="both", expand=True)

# Tạo bảng dữ liệu
columns = ("ID", "Biển Số", "Thời Gian")
tree = ttk.Treeview(table_frame, 
                   columns=columns, 
                   show="headings", 
                   height=8,
                   style="Custom.Treeview")

tree.column("ID", anchor="center", width=80)
tree.column("Biển Số", anchor="center", width=200)
tree.column("Thời Gian", anchor="center", width=250)

tree.heading("ID", text="ID", anchor="center")
tree.heading("Biển Số", text="BIỂN SỐ", anchor="center")
tree.heading("Thời Gian", text="THỜI GIAN", anchor="center")

scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscroll=scroll_y.set)
scroll_y.pack(side="right", fill="y")

tree.pack(fill="both", expand=True)
update_table()

root.mainloop()
