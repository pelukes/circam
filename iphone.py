import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI(title="iPhone CIR Generator - Optimized")

def process_cir(rgb_bytes, ir_bytes):
    # 1. Načtení dat
    img_rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_ir = cv2.imdecode(np.frombuffer(ir_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img_rgb is None or img_ir is None:
        raise ValueError("Chyba dekódování.")

    h, w = img_rgb.shape[:2]
    
    # Pro registraci použijeme pracovní rozlišení (např. 1200px šířka pro přesnost)
    work_w = 1200
    work_scale = work_w / float(w)
    work_h = int(h * work_scale)
    
    gray_rgb = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), (work_w, work_h))
    gray_ir = cv2.resize(cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY), (work_w, work_h))

    # --- VYLEPŠENÁ REGISTRACE POMOCÍ AKAZE ---
    # AKAZE je skvělý pro multispektrální data, protože je invariantní k jasu
    detector = cv2.AKAZE_create()
    kp1, desc1 = detector.detectAndCompute(gray_rgb, None)
    kp2, desc2 = detector.detectAndCompute(gray_ir, None)

    # Matchování bodů
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2)

    # Výběr nejlepších shod (top 20%)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.2)]

    # Extrakce souřadnic bodů
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Výpočet homografie (matice 3x3 pro perspektivní transformaci)
    # RANSAC odfiltruje špatně spárované body
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # --- FINÁLNÍ SLOŽENÍ ---
    # Pro finální obraz použijeme měřítko, které chcete (např. 0.7 pro zachování detailů)
    out_scale = 0.7
    out_w, out_h = int(w * out_scale), int(h * out_scale)

    # Přepočet matice M pro cílové rozlišení
    # M_scale = S_out * M * S_work_inv
    S_work_inv = np.diag([1/work_scale, 1/work_scale, 1])
    S_out = np.diag([out_scale, out_scale, 1])
    M_final = S_out @ S_work_inv @ M @ S_work_inv.copy() # Zjednodušený odhad

    img_rgb_res = cv2.resize(img_rgb, (out_w, out_h))
    img_ir_res = cv2.resize(img_ir, (out_w, out_h))

    # Aplikace perspektivní transformace na IR snímek
    img_ir_aligned = cv2.warpPerspective(img_ir_res, M_final, (out_w, out_h))

    # Skládání podle vašeho požadavku: RED=NIR, GREEN=G_orig, BLUE=B_orig
    # V OpenCV BGR: [B, G, R]
    nir = img_ir_aligned[:, :, 2]
    green = img_rgb_res[:, :, 1]
    blue = img_rgb_res[:, :, 0]

    cir_composite = np.dstack((blue, green, nir))

    is_success, buffer = cv2.imencode(".jpg", cir_composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return io.BytesIO(buffer)

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIR Fast Generator</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, sans-serif; text-align: center; padding: 20px; background: #1a1a1a; color: white; }
            .card { background: #2a2a2a; padding: 20px; border-radius: 15px; max-width: 500px; margin: auto; }
            input { margin: 15px 0; display: block; width: 100%; }
            button { background: #34c759; color: white; border: none; padding: 15px; border-radius: 10px; width: 100%; font-weight: bold; }
            #result img { max-width: 100%; margin-top: 20px; border-radius: 10px; border: 1px solid #444; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>CIR Generator 🛰️</h2>
            <form id="uploadForm">
                <label>1. Standardní RGB (s filtrem):</label>
                <input type="file" id="rgbFile" accept="image/*" required>
                <label>2. IR fotka (bez filtru):</label>
                <input type="file" id="irFile" accept="image/*" required>
                <button type="submit">ZPRACOVAT</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const btn = e.target.querySelector('button');
                const resDiv = document.getElementById('result');
                btn.innerText = 'Běží registrace...';
                
                const formData = new FormData();
                formData.append('file_rgb', document.getElementById('rgbFile').files[0]);
                formData.append('file_ir', document.getElementById('irFile').files[0]);

                const response = await fetch('/process/', { method: 'POST', body: formData });
                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    resDiv.innerHTML = `<h3>Hotovo:</h3><img src="${url}"><br><a href="${url}" download="cir_result.jpg" style="color:#34c759">Uložit výsledek</a>`;
                } else {
                    alert('Chyba. Zkuste menší soubory nebo restartovat server.');
                }
                btn.innerText = 'ZPRACOVAT DALŠÍ';
            };
        </script>
    </body>
    </html>
    """

@app.post("/process/")
async def upload_images(file_rgb: UploadFile = File(...), file_ir: UploadFile = File(...)):
    rgb_data = await file_rgb.read()
    ir_data = await file_ir.read()
    try:
        result_buffer = process_cir(rgb_data, ir_data)
        return StreamingResponse(result_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
