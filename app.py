from flask import Flask, render_template, request, jsonify
import subprocess
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ochrana pro běh na serveru bez GUI
import matplotlib.pyplot as plt
from PIL import Image
import gc
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

# Cesty pro dočasné soubory v RAM (RAM disk Linuxu) - extrémně zrychluje I/O operace
RAM_RGB_PATH = "/dev/shm/rgb.jpg"
RAM_NIR_PATH = "/dev/shm/nir.jpg"

# Cesty pro finální soubory servírované webovým serverem
STATIC_RGB_PATH = "static/rgb.jpg"
STATIC_NIR_PATH = "static/nir.jpg"
CIR_PATH = "static/cir.jpg"
NDVI_PATH = "static/ndvi.jpg"

def capture_image(filepath):
    # Uzamčení AWB a expozice pro vědeckou konzistenci.
    cmd = ["rpicam-still", "-t", "1000", "--width", "1600", "--height", "1200", "--nopreview", "-o", filepath]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Chyba kamery: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture/<img_type>', methods=['POST'])
def capture(img_type):
    if img_type == 'rgb':
        success = capture_image(RAM_RGB_PATH)
        if success:
            shutil.copy2(RAM_RGB_PATH, STATIC_RGB_PATH) # Kopie z RAM na SD pro web
            
    elif img_type == 'nir':
        success = capture_image(RAM_NIR_PATH)
        if success:
            try:
                # Zpracování NIR přímo v RAM
                with Image.open(RAM_NIR_PATH) as img:
                    nir_grayscale = img.split()[0].convert('L')
                    nir_grayscale.save(RAM_NIR_PATH, quality=95)
                
                # Přesun upraveného souboru z RAM na web
                shutil.copy2(RAM_NIR_PATH, STATIC_NIR_PATH)
                
                del nir_grayscale
                gc.collect()
            except Exception as e:
                print(f"Chyba při zpracování NIR: {e}")
    else:
        return jsonify({"error": "Neplatný typ snímku"}), 400

    if success:
        return jsonify({"status": "success", "message": f"Snímek {img_type.upper()} byl úspěšně pořízen."})
    else:
        return jsonify({"error": "Chyba při pořizování snímku"}), 500

@app.route('/generate_cir', methods=['POST'])
def generate_cir():
    if not os.path.exists(RAM_RGB_PATH) or not os.path.exists(RAM_NIR_PATH):
        return jsonify({"error": "Nejprve je nutné pořídit RGB i NIR snímky."}), 400

    try:
        # Optimalizované načítání z RAM disku
        rgb_img = Image.open(RAM_RGB_PATH)
        rgb_img.draft('RGB', rgb_img.size) # Rychlejší dekódování JPEGu
        rgb_img = rgb_img.convert('RGB')
        
        nir_img = Image.open(RAM_NIR_PATH)
        nir_img.draft('L', nir_img.size)
        nir_grayscale = nir_img.convert('L')

        _, g, b = rgb_img.split()
        
        # CIR Kompozit: Červený kanál obrazu = NIR senzorová data, Zelený = Zelený, Modrý = Modrý
        cir_img = Image.merge('RGB', (nir_grayscale, g, b))
        
        rgb_img.close()
        nir_img.close()
        del rgb_img, nir_img, g, b, nir_grayscale
        gc.collect()

        cir_img.save(CIR_PATH, quality=90)
        del cir_img
        gc.collect()
        
        return jsonify({"status": "success", "message": "CIR kompozit byl rychle vygenerován."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_ndvi', methods=['POST'])
def generate_ndvi():
    if not os.path.exists(RAM_RGB_PATH) or not os.path.exists(RAM_NIR_PATH):
        return jsonify({"error": "Nejprve je nutné pořídit RGB i NIR snímky."}), 400

    try:
        # Čtení přímo z RAM disku
        with Image.open(RAM_RGB_PATH) as rgb_img:
            red_band_img = rgb_img.split()[0]
            
        with Image.open(RAM_NIR_PATH) as nir_img:
            nir_band_img = nir_img.split()[0]

        red_band = np.array(red_band_img, dtype=np.float32)
        nir_band = np.array(nir_band_img, dtype=np.float32)
        
        del red_band_img, nir_band_img
        gc.collect()

        # IN-PLACE operace NumPy pro masivní úsporu paměti a CPU času
        # 1. Jmenovatel (NIR + Red)
        denominator = np.add(nir_band, red_band)
        np.maximum(denominator, 0.0001, out=denominator) # Zabránění dělení nulou
        
        # 2. Čitatel a rovnou finální NDVI (recyklujeme proměnnou nir_band)
        np.subtract(nir_band, red_band, out=nir_band) # Nyní nir_band = (NIR - Red)
        np.divide(nir_band, denominator, out=nir_band) # Nyní nir_band = NDVI
        
        # Statistický stretching
        vmin = max(float(np.percentile(nir_band, 2)), -1.0)
        vmax = min(float(np.percentile(nir_band, 98)), 1.0)

        # Přímý zápis rastru do souboru (obchází pomalý vektorový engine Matplotlibu)
        plt.imsave(NDVI_PATH, nir_band, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        
        del red_band, denominator, nir_band
        gc.collect()
        
        return jsonify({"status": "success", "message": f"NDVI vygenerováno extrémně rychle (rozsah: {vmin:.2f} až {vmax:.2f})."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    recipient = data.get('email')
    
    if not recipient:
        return jsonify({"error": "Není zadána e-mailová adresa."}), 400

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465
    SENDER_EMAIL = "pe.lukes@gmail.com"
    SENDER_PASSWORD = "endrcqxdassqmuck" 

    msg = EmailMessage()
    msg['Subject'] = 'Multispektrální Snímky (CzechGlobe)'
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient
    msg.set_content('V příloze naleznete multispektrální snímky pořízené systémem Raspberry Pi.\nPoznámka: NDVI snímek využívá dynamický statistický stretch (vizualizace 2. až 98. percentilu).')

    images = [STATIC_RGB_PATH, STATIC_NIR_PATH, CIR_PATH, NDVI_PATH]
    attached_count = 0
    
    for img_path in images:
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                img_data = f.read()
                filename = os.path.basename(img_path)
                # Přejmenování souborů z RAM při odesílání, aby nedošlo ke zmatení u statických cest
                if "static/" in filename:
                    filename = filename.replace("static/", "")
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=filename)
                attached_count += 1

    if attached_count == 0:
        return jsonify({"error": "Neexistují žádné snímky k odeslání."}), 400

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return jsonify({"status": "success", "message": f"Snímky ({attached_count}) odeslány na {recipient}."})
    except Exception as e:
        return jsonify({"error": f"Chyba SMTP serveru: {str(e)}"}), 500

if __name__ == '__main__':
    # Ujistěte se, že statická složka existuje
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=False)
