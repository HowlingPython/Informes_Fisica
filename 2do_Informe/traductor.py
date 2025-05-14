"""
Convierte todos los archivos .mov de un directorio de entrada
a .mp4 con calidad máxima (qscale 0) en el directorio de salida,
mostrando solo una barra de progreso.
"""

import subprocess
import os
from glob import glob
from tqdm import tqdm  # pip install tqdm

def convert_mov_to_mp4(input_path: str, output_path: str):
    """
    Convierte silenciosamente un .mov a .mp4 sin pérdida.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",        # quita la cabecera de ffmpeg
        "-loglevel", "error",  # solo muestra errores graves
        "-y",                  # sobrescribe sin preguntar
        "-i", input_path,
        "-q:v", "0",
        output_path
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,  # suprime stdout
        stderr=subprocess.DEVNULL,  # suprime stderr
        check=True
    )

def batch_convert(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    mov_files = glob(os.path.join(input_dir, "*.mov"))
    if not mov_files:
        print(f"No se encontraron .mov en {input_dir}")
        return

    # tqdm mostrará: Convirtiendo:  30%|███       | 3/10 [00:12<00:30, 0.23vídeo/s]
    for mov in tqdm(mov_files, desc="Convirtiendo", unit="video"):
        base = os.path.splitext(os.path.basename(mov))[0]
        mp4  = os.path.join(output_dir, f"{base}.mp4")
        convert_mov_to_mp4(mov, mp4)

    print("¡Conversión completada!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convierte vídeos .mov a .mp4 mostrando solo progreso."
    )
    parser.add_argument(
        "input_dir",
        help="Carpeta donde están tus .mov (p. ej. raw_videos/)"
    )
    parser.add_argument(
        "output_dir",
        help="Carpeta donde guardar los .mp4 convertidos (p. ej. converted_videos/)"
    )
    args = parser.parse_args()
    batch_convert(args.input_dir, args.output_dir)
