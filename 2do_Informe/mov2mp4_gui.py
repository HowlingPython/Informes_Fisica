import logging
import os
import platform
import subprocess
import sys
import threading
import queue
import shutil
import tkinter as tk
from pathlib import Path
from dotenv import load_dotenv
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
DEFAULT_CRF = os.environ.get("DEFAULT_CRF", "18")
DEFAULT_PRESET = os.environ.get("DEFAULT_PRESET", "medium")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3"))

LOG_FILE = Path.home() / ".mov2mp4_converter.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(message)s"
)

# ─── Conversion ────────────────────────────────────────────────────────────────

def convert_mov_to_mp4(input_path: Path, output_dir: Path, crf: str, preset: str):
    try:
        base = input_path.stem
        out_file = output_dir / f"{base}.mp4"
        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-crf", crf,
            "-preset", preset,
            str(out_file)
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if platform.system()=="Windows" else 0
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,
                       check=True,
                       creationflags=creationflags)
        return (input_path, True, "")
    except subprocess.CalledProcessError as e:
        logging.error("Conversion failed for %s: %s", input_path, e.stderr.decode(errors="ignore"))
        return (input_path, False, e.stderr.decode(errors="ignore"))

# ─── Conversion Runner ─────────────────────────────────────────────────────────

def run_conversion(files, out_dir, crf, preset, progress_queue, cancel_event):
    total = len(files)
    completed = 0
    batch_size = BATCH_SIZE  # Limita la cantidad de procesos simultáneos

    for i in range(0, total, batch_size):
        if cancel_event.is_set():
            break
        batch = files[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as execr:
            futures = [
                execr.submit(convert_mov_to_mp4, Path(f), Path(out_dir), crf, preset)
                for f in batch
            ]
            for future in futures:
                if cancel_event.is_set():
                    break
                inp, success, err = future.result()
                completed += 1
                progress = completed / total * 100
                progress_queue.put(("progress", progress))
                progress_queue.put(("status", f"{completed}/{total}"))

    if cancel_event.is_set():
        progress_queue.put(("done", ("info", "Conversión cancelada.")))
    else:
        progress_queue.put(("done", ("info", "Conversión completada. Abrir carpeta de salida?")))

# ─── GUI ────────────────────────────────────────────────────────────────────────

class ConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MOV → MP4 Converter")
        self.resizable(False, False)
        self.geometry("420x200")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Listo.")
        self.progress = ttk.Progressbar(self, orient="horizontal", length=380, mode="determinate")
        self.queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.crf_value = DEFAULT_CRF
        self.preset_value = DEFAULT_PRESET

        self.btn_quality = tk.Button(self, text="Ajustes de calidad", command=self.show_quality_settings)
        self.btn_quality.pack(pady=(10, 5))

        self.btn_convert = tk.Button(self, text="Seleccionar y convertir", command=self.start)
        self.btn_convert.pack(pady=5)
        self.btn_cancel = tk.Button(self, text="Cancelar", state="disabled", command=self.cancel)
        self.btn_cancel.pack()

        self.progress.pack(pady=(15,5))
        tk.Label(self, textvariable=self.status_var).pack()

    def show_quality_settings(self):
        top = tk.Toplevel(self)
        top.title("Ajustes de calidad")
        top.geometry("300x100")
        top.resizable(False, False)

        frame = tk.Frame(top)
        frame.pack(pady=10)

        tk.Label(frame, text="CRF").grid(row=0, column=0)
        crf_entry = tk.Entry(frame, width=5)
        crf_entry.insert(0, self.crf_value)
        crf_entry.grid(row=0, column=1, padx=(0,10))
        crf_entry.bind("<Enter>", lambda e: self.status_var.set("CRF: Calidad del video (18-28, menor es mejor)"))
        crf_entry.bind("<Leave>", lambda e: self.status_var.set("Listo."))

        tk.Label(frame, text="Preset").grid(row=0, column=2)
        preset_combo = ttk.Combobox(frame, values=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], width=10)
        preset_combo.set(self.preset_value)
        preset_combo.grid(row=0, column=3)
        preset_combo.bind("<Enter>", lambda e: self.status_var.set("Preset: Velocidad vs compresión"))
        preset_combo.bind("<Leave>", lambda e: self.status_var.set("Listo."))

        def apply():
            self.crf_value = crf_entry.get()
            self.preset_value = preset_combo.get()
            top.destroy()

        tk.Button(top, text="Aplicar", command=apply).pack(pady=5)

    def start(self):
        files = filedialog.askopenfilenames(filetypes=[("MOV", "*.mov")], title="Seleccionar archivos .mov")
        if not files: return
        out_dir = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if not out_dir: return

        self.out_dir = out_dir
        self.btn_convert.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress["value"] = 0
        self.status_var.set("Iniciando...")

        self.cancel_event.clear()
        threading.Thread(
            target=run_conversion,
            args=(files, out_dir,
                  self.crf_value,
                  self.preset_value,
                  self.queue,
                  self.cancel_event),
            daemon=True
        ).start()
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                kind, val = self.queue.get_nowait()
                if kind == "progress":
                    self.progress["value"] = val
                elif kind == "status":
                    self.status_var.set(f"{val}")
                elif kind == "done":
                    level, msg = val
                    self.btn_cancel.config(state="disabled")
                    if level == "info":
                        if "Abrir carpeta" in msg:
                            if messagebox.askyesno("¡Listo!", f"{msg}"):
                                if platform.system()=="Windows":
                                    os.startfile(self.out_dir)
                                elif platform.system()=="Darwin":
                                    subprocess.run(["open", self.out_dir])
                                else:
                                    subprocess.run(["xdg-open", self.out_dir])
                        else:
                            messagebox.showinfo("Info", msg)
                    else:
                        messagebox.showwarning("Aviso", msg)
                    self.btn_convert.config(state="normal")
                    return
        except queue.Empty:
            self.after(100, self._poll_queue)

    def cancel(self):
        self.cancel_event.set()
        self.status_var.set("Cancelando...")

    def on_close(self):
        if self.btn_cancel["state"] == "normal" and not self.cancel_event.is_set():
            if not messagebox.askokcancel("Confirmar", "¿Cancelar conversión y salir?"):
                return
        self.destroy()
        sys.exit(0)

# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    if not shutil.which(FFMPEG_BIN):
        tk.Tk().withdraw()
        messagebox.showerror("Error", f"No se encontró '{FFMPEG_BIN}' en PATH.")
        sys.exit(1)

    app = ConverterGUI()
    app.mainloop()

if __name__ == "__main__":
    main()


"""
Para copilar usar:

pyinstaller --onefile --windowed --noconfirm `
  --name mov2mp4_gui `
  --add-data ".env;." `
  mov2mp4_gui.py

"""
