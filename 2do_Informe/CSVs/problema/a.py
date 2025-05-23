import glob
import os

def is_header(parts):
    # la cabecera (p.ej. 't,x,y') o filas inesperadas las dejamos
    return not parts[0].lstrip('-').isdigit()

def process_line(line):
    line = line.rstrip('\n')
    parts = line.split(',')
    # si no son exactamente 6 partes (3 números “cortados” por coma decimal), devuelvo línea original
    if is_header(parts) or len(parts) != 6:
        return line
    # reconstruyo cada número: par[0].par[1], par[2].par[3], par[4].par[5]
    t = parts[0] + '.' + parts[1]
    x = parts[2] + '.' + parts[3]
    y = parts[4] + '.' + parts[5]
    return ','.join((t, x, y))

def process_file(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as f_in, \
         open(outfile, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(process_line(line) + '\n')
    print(f"{infile} → {outfile}")

def main():
    # busca todos los .csv del directorio actual
    csv_files = glob.glob('*.csv')
    if not csv_files:
        print("No hay archivos .csv en esta carpeta.")
        return

    for inp in csv_files:
        # evita procesar de nuevo los ya “fixed”
        if inp.endswith('_fixed.csv'):
            continue
        base, ext = os.path.splitext(inp)
        outp = f"{base}_fixed{ext}"
        process_file(inp, outp)

if __name__ == "__main__":
    main()
