import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="iPhone CIR & GNDVI Generator", layout="centered")

def process_images(rgb_img, ir_img):
    # Převod z PIL na OpenCV (BGR)
    img_rgb = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    img_ir = cv2.cvtColor(np.array(ir_img), cv2.COLOR_RGB2BGR)

    h, w = img_rgb.shape[:2]
    
    # 1. Příprava pro registraci
    work_w = 1200
    work_scale = work_w / float(w)
    work_h = int(h * work_scale)
    
    gray_rgb = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), (work_w, work_h))
    gray_ir = cv2.resize(cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY), (work_w, work_h))

    # 2. Registrace (AKAZE)
    detector = cv2.AKAZE_create()
    kp1, desc1 = detector.detectAndCompute(gray_rgb, None)
    kp2, desc2 = detector.detectAndCompute(gray_ir, None)

    if desc1 is None or desc2 is None:
        return None, None, "Nepodařilo se nalézt identifikační body."

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.2)]

    if len(good_matches) < 4:
        return None, None, "Nedostatečný počet shod pro registraci."

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 3. Finální měřítko (0.7)
    out_scale = 0.7
    out_w, out_h = int(w * out_scale), int(h * out_scale)
    
    # Korekce matice pro změnu rozlišení
    S_work_inv = np.diag([1/work_scale, 1/work_scale, 1])
    S_out = np.diag([out_scale, out_scale, 1])
    M_final = S_out @ S_work_inv @ M @ S_work_inv

    img_rgb_res = cv2.resize(img_rgb, (out_w, out_h))
    img_ir_res = cv2.resize(img_ir, (out_w, out_h))
    img_ir_aligned = cv2.warpPerspective(img_ir_res, M_final, (out_w, out_h))

    # --- KANÁLY ---
    # OpenCV BGR -> nir:2, green:1, blue:0
    nir = img_ir_aligned[:, :, 2].astype(float)
    green = img_rgb_res[:, :, 1].astype(float)
    blue = img_rgb_res[:, :, 0].astype(float)

    # 4. CIR Kompozit (R=NIR, G=Green, B=Blue)
    cir_composite = np.dstack((blue, green, nir)).astype(np.uint8)
    cir_rgb = cv2.cvtColor(cir_composite, cv2.COLOR_BGR2RGB)

    # 5. GNDVI Výpočet
    # Přidáme malou konstantu (epsilon), abychom se vyhnuli dělení nulou
    gndvi = (nir - green) / (nir + green + 1e-8)
    
    # Škálování GNDVI pro vizualizaci (z -1.0..1.0 na 0..255)
    # Většinou nás zajímá rozsah 0 až 1 pro vegetaci
    gndvi_scaled = np.clip(gndvi, 0, 1) 
    gndvi_8bit = (gndvi_scaled * 255).astype(np.uint8)
    
    # Aplikace barevné mapy (ColorMap)
    gndvi_colored = cv2.applyColorMap(gndvi_8bit, cv2.COLORMAP_SUMMER) # SUMMER je žluto-zelená
    gndvi_colored_rgb = cv2.cvtColor(gndvi_colored, cv2.COLOR_BGR2RGB)

    return cir_rgb, gndvi_colored_rgb, None

# --- UI ---
st.title("🛰️ Remote Sensing: CIR & GNDVI")
st.write("Generátor Color-InfraRed kompozitu a GNDVI indexu pro monitoring vegetace.")

col1, col2 = st.columns(2)
with col1:
    rgb_file = st.file_uploader("1. RGB Snímek", type=['jpg', 'jpeg', 'png'])
with col2:
    ir_file = st.file_uploader("2. IR Snímek (bez filtru)", type=['jpg', 'jpeg', 'png'])

if rgb_file and ir_file:
    if st.button("SPOČÍTAT ANALÝZU"):
        with st.spinner("Registruji snímky a počítám indexy..."):
            try:
                img_rgb = Image.open(rgb_file)
                img_ir = Image.open(ir_file)
                
                cir_res, gndvi_res, error = process_images(img_rgb, img_ir)
                
                if error:
                    st.error(error)
                else:
                    st.subheader("1. Color Infrared (CIR)")
                    st.image(cir_res, caption="NIR -> Red channel", use_container_width=True)
                    
                    st.subheader("2. Green NDVI (GNDVI)")
                    st.image(gndvi_res, caption="Index (NIR-G)/(NIR+G) - Barevná škála Summer", use_container_width=True)
                    
                    # Download tlačítka
                    buf_cir = io.BytesIO()
                    Image.fromarray(cir_res).save(buf_cir, format="JPEG")
                    
                    buf_gndvi = io.BytesIO()
                    Image.fromarray(gndvi_res).save(buf_gndvi, format="JPEG")

                    c1, c2 = st.columns(2)
                    c1.download_button("Stáhnout CIR", buf_cir.getvalue(), "cir.jpg", "image/jpeg")
                    c2.download_button("Stáhnout GNDVI", buf_gndvi.getvalue(), "gndvi.jpg", "image/jpeg")

            except Exception as e:
                st.error(f"Chyba: {e}")
