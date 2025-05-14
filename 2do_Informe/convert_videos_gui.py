import os
import sys
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

def convert_mov_to_mp4(input_path: str, output: str):
    try:
        if os.path.isdir(output):
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output, f"{base}.mp4")
        else:
            output_path = output

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", input_path,
            "-q:v", "0",
            output_path
        ]
        
        creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL, check=True,
               creationflags=creationflags)
        
        return (input_path, "Success")
    except subprocess.CalledProcessError as e:
        return (input_path, f"Failed: {e}")

def run_conversion_in_parallel(files, out_dir, root, progress_bar, status_var, queue_results):
    results = []
    total = len(files)
    completed = 0
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(convert_mov_to_mp4, mov, out_dir): mov for mov in files}
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            completed += 1
            queue_results.put(('progress', completed / total * 100))
            queue_results.put(('status', f"Procesando {completed}/{total}"))

    failed = [f for f, status in results if status != "Success"]
    if failed:
        queue_results.put(('done', ("warn", f"Fallaron {len(failed)} archivos:\n" + "\n".join(failed))))
    else:
        queue_results.put(('done', ("info", "Conversión completada.")))

def start_conversion(root, progress_bar, status_var):
    files = filedialog.askopenfilenames(
        title="Seleccionar archivos .mov",
        filetypes=[("MOV files", "*.mov")],
    )
    if not files:
        return

    out_dir = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    if not out_dir:
        return

    # Disable button during conversion
    for widget in root.winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(state="disabled")

    queue_results = queue.Queue()
    threading.Thread(
        target=run_conversion_in_parallel,
        args=(files, out_dir, root, progress_bar, status_var, queue_results),
        daemon=True
    ).start()

    def update_ui():
        try:
            while True:
                item = queue_results.get_nowait()
                if item[0] == 'progress':
                    progress_bar['value'] = item[1]
                elif item[0] == 'status':
                    status_var.set(item[1])
                elif item[0] == 'done':
                    msg_type, msg_text = item[1]
                    if msg_type == "warn":
                        messagebox.showwarning("Conversión incompleta", msg_text)
                    else:
                        messagebox.showinfo("¡Listo!", msg_text)
                    root.quit()
                    sys.exit(0)
        except queue.Empty:
            root.after(100, update_ui)

    update_ui()

def main():
    root = tk.Tk()
    root.title("MOV → MP4 Converter")
    root.geometry("400x180")
    root.resizable(False, False)

    status_var = tk.StringVar(value="Listo para comenzar...")

    btn = tk.Button(
        root,
        text="Seleccionar y convertir",
        command=lambda: start_conversion(root, progress, status_var),
        width=30,
        height=2
    )
    btn.pack(pady=15)

    progress = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate")
    progress.pack(pady=10)

    status_label = tk.Label(root, textvariable=status_var)
    status_label.pack()

    root.mainloop()

def run_gui():
    try:
        main()
    except Exception as e:
        tk.Tk().withdraw()
        messagebox.showerror("Error inesperado", str(e))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Necesario para pyinstaller + Windows
    run_gui()


"""
Para copilar usar:

pyinstaller --onefile `
  --windowed `
  --noconfirm `
  --name convertir_videos_gui `
  --add-binary "C:\tu\path\a\bin\ffmpeg.exe;." `
  convert_videos_gui.py

"""