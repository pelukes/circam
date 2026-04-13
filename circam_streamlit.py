import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

st.set_page_config(page_title="iPhone CIR Generator", layout="centered")

def process_cir(rgb_img, ir_img):
    # Převod z PIL (Streamlit default) na OpenCV (BGR)
    img_rgb = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    img_ir = cv2.cvtColor(np.array(ir_img), cv2.COLOR_RGB2BGR)

    h, w = img_rgb.shape[:2]
    
    # 1. Příprava pro registraci (pracovní rozlišení)
    work_w = 1200
    work_scale = work_w / float(w)
    work_h = int(h * work_scale)
    
    gray_rgb = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), (work_w, work_h))
    gray_ir = cv2.resize(cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY), (work_w, work_h))

    # 2. VYLEPŠENÁ REGISTRACE (AKAZE)
    detector = cv2.AKAZE_create()
    kp1, desc1 = detector.detectAndCompute(gray_rgb, None)
    kp2, desc2 = detector.detectAndCompute(gray_ir, None)

    if desc1 is None or desc2 is None:
        return None, "Nepodařilo se nalézt dostatek identifikačních bodů."

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.2)]

    if len(good_matches) < 4:
        return None, "Nedostatečný počet shod pro registraci."

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Výpočet homografie
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 3. Finální měřítko výstupu (0.7 pro rychlost a kvalitu)
    out_scale = 0.7
    out_w, out_h = int(w * out_scale), int(h * out_scale)

    # Přepočet matice pro nové rozlišení
    S_work_inv = np.diag([1/work_scale, 1/work_scale, 1])
    S_out = np.diag([out_scale, out_scale, 1])
    M_final = S_out @ S_work_inv @ M @ S_work_inv

    img_rgb_res = cv2.resize(img_rgb, (out_w, out_h))
    img_ir_res = cv2.resize(img_ir, (out_w, out_h))

    # Aplikace registrace na IR snímek
    img_ir_aligned = cv2.warpPerspective(img_ir_res, M_final, (out_w, out_h))

    # 4. Barevná syntéza: RED = NIR, GREEN = G_orig, BLUE = B_orig
    # OpenCV BGR: [0:Blue, 1:Green, 2:Red]
    nir = img_ir_aligned[:, :, 2]
    green = img_rgb_res[:, :, 1]
    blue = img_rgb_res[:, :, 0]

    cir_composite = np.dstack((blue, green, nir))
    
    # Převod zpět na RGB pro zobrazení v Streamlitu
    return cv2.cvtColor(cir_composite, cv2.COLOR_BGR2RGB), None

# --- UI STRÁNKY ---
st.title("🛰️ iPhone CIR Generator")
st.write("Vytvořte Color-InfraRed kompozit ze dvou fotografií.")

col1, col2 = st.columns(2)
with col1:
    rgb_file = st.file_uploader("1. Standardní RGB (s filtrem)", type=['jpg', 'jpeg', 'png', 'dng'])
with col2:
    ir_file = st.file_uploader("2. IR snímek (bez filtru)", type=['jpg', 'jpeg', 'png', 'dng'])

if rgb_file and ir_file:
    if st.button("GENEROVAT CIR KOMPOZIT"):
        with st.spinner("Probíhá registrace obrazu a skládání kanálů..."):
            try:
                # Načtení do PIL
                img_rgb = Image.open(rgb_file)
                img_ir = Image.open(ir_file)
                
                # Zpracování
                result, error = process_cir(img_rgb, img_ir)
                
                if error:
                    st.error(error)
                else:
                    st.success("Hotovo!")
                    st.image(result, caption="Výsledný CIR kompozit (RED=NIR)", use_container_width=True)
                    
                    # Příprava ke stažení
                    result_pil = Image.fromarray(result)
                    buf = io.BytesIO()
                    result_pil.save(buf, format="JPEG", quality=90)
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Stáhnout výsledek",
                        data=byte_im,
                        file_name="cir_composite.jpg",
                        mime="image/jpeg"
                    )
            except Exception as e:
                st.error(f"Došlo k chybě: {e}")
else:
    st.info("Nahrajte oba soubory pro spuštění zpracování.")
