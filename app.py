from flask import Flask, request, jsonify
import cv2
import numpy as np
import keras
import tensorflow as tf
import base64
import os

app = Flask(__name__)

model = keras.models.load_model(
    "model_telur.keras",
    compile=False
)

# ── WARM-UP: paksa model "terbangun" sepenuhnya sebelum dipakai ──
_dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
_ = model(_dummy_input)
print("[STARTUP] Model warmed up, input shape defined.")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image_from_array(img):
    h_orig, w_orig = img.shape[:2]

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # STEP 1 — DETEKSI & CROP TELUR
    scale = 600 / max(h_orig, w_orig)
    img_small = cv2.resize(img, (int(w_orig*scale), int(h_orig*scale)))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)
    edges = cv2.Canny(blur, 30, 120)
    kernel = np.ones((9,9), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crop = img.copy()
    if contours:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        h_s, w_s = img_small.shape[:2]
        img_small_area = h_s * w_s
        for c in contours_sorted:
            area = cv2.contourArea(c)
            if area < 0.05*img_small_area or area > 0.95*img_small_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if max(w,h) / (min(w,h) + 1e-5) > 3.0:
                continue
            x,y,w,h = int(x/scale),int(y/scale),int(w/scale),int(h/scale)
            pad_x, pad_y = int(w*0.08), int(h*0.08)
            x1,y1 = max(0,x-pad_x), max(0,y-pad_y)
            x2,y2 = min(w_orig,x+w+pad_x), min(h_orig,y+h+pad_y)
            crop = img[y1:y2, x1:x2]
            break

    # STEP 2 — RESIZE
    resized = cv2.resize(crop, (224, 224))

    # STEP 3 — GRAYSCALE
    gray_final = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # STEP 4 — CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16,16))
    enhanced = clahe.apply(gray_final)

    # STEP 5 — SHARPENING
    kernel_sharpen = np.array([
        [ 0, -0.5,  0],
        [-0.5, 3.0, -0.5],
        [ 0, -0.5,  0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # STEP 6 — BACK TO RGB
    final = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return final


# ─────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────
def generate_gradcam(processed_input, original_display):
    try:
        # Pisahkan base_model (MobileNetV2) dan head_layers (GAP, Dense, Dropout, Dense_1)
        base_model = None
        head_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
            else:
                head_layers.append(layer)

        if base_model is None:
            print("[GradCAM] Base model tidak ditemukan, skip")
            return None

        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            print("[GradCAM] Conv layer tidak ditemukan di base_model, skip")
            return None

        print(f"[GradCAM] Menggunakan layer: {last_conv_layer} (di dalam {base_model.name})")

        # PENTING: grad_model cukup sampai base_model saja
        # base_model.input SUDAH pasti terdefinisi karena dia model Functional mandiri
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer(last_conv_layer).output,
                base_model.output
            ]
        )

        input_tensor = np.expand_dims(processed_input, axis=0)
        input_tensor = tf.cast(input_tensor, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, base_features = grad_model(input_tensor)

            # Lanjutkan manual lewat head layers (GAP -> Dense -> Dropout -> Dense_1)
            x = base_features
            for layer in head_layers:
                x = layer(x)
            predictions = x
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = heatmap.numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_display, 0.55, heatmap_colored, 0.45, 0)

        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        return None


# ─────────────────────────────────────────────
# DETEKSI TELUR (tidak berubah)
# ─────────────────────────────────────────────
def detect_eggs_from_array(image):
    h_img, w_img = image.shape[:2]
    scale = 1200 / max(h_img, w_img)
    img_small = cv2.resize(image, (int(w_img*scale), int(h_img*scale)))
    h_s, w_s = img_small.shape[:2]

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    egg_mask = cv2.inRange(
        hsv, np.array([0,60,80]), np.array([30,255,255])
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    egg_mask = cv2.morphologyEx(egg_mask, cv2.MORPH_CLOSE, kernel)
    egg_mask = cv2.morphologyEx(egg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        egg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if cv2.contourArea(c) > h_s*w_s*0.005]

    print(f"[DEBUG] img size: {w_img}x{h_img}")
    print(f"[DEBUG] egg colored regions: {len(valid)}")

    if valid:
        all_pts = np.vstack(valid)
        tx, ty, tw, th = cv2.boundingRect(all_pts)
        pad = 20
        tx = max(0,tx-pad); ty = max(0,ty-pad)
        tx2 = min(w_s,tx+tw+pad*2); ty2 = min(h_s,ty+th+pad*2)

        tray_area_ratio = ((tx2-tx)*(ty2-ty)) / (h_s*w_s)
        print(f"[DEBUG] tray_area_ratio: {tray_area_ratio:.3f}")

        if tray_area_ratio > 0.10:
            tray_crop = img_small[ty:ty2, tx:tx2]
            tc_w = tray_crop.shape[1]

            gray_tray = cv2.cvtColor(tray_crop, cv2.COLOR_BGR2GRAY)
            blur_tray = cv2.GaussianBlur(gray_tray, (9,9), 2)

            r_min = int(tc_w/16)
            r_max = int(tc_w/8)

            circles = cv2.HoughCircles(
                blur_tray, cv2.HOUGH_GRADIENT,
                dp=1.0, minDist=int(r_min*1.4),
                param1=80, param2=25,
                minRadius=r_min, maxRadius=r_max
            )

            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                print(f"[DEBUG] HoughCircles raw: {len(circles)}")

                tray_egg_mask = egg_mask[ty:ty2, tx:tx2]
                filtered = []
                for (cx, cy, cr) in circles:
                    if (0<=cy<tray_egg_mask.shape[0] and
                            0<=cx<tray_egg_mask.shape[1]):
                        y1=max(0,cy-10)
                        y2_=min(tray_egg_mask.shape[0],cy+10)
                        x1=max(0,cx-10)
                        x2_=min(tray_egg_mask.shape[1],cx+10)
                        if np.mean(tray_egg_mask[y1:y2_,x1:x2_]) > 30:
                            filtered.append((cx,cy,cr))

                print(f"[DEBUG] filtered: {len(filtered)}")

                egg_crops = []
                for (cx, cy, cr) in filtered:
                    abs_x = (tx+cx)/scale
                    abs_y = (ty+cy)/scale
                    abs_r = cr/scale
                    shrink = int(abs_r*0.05)
                    x1=max(0,int(abs_x-abs_r+shrink))
                    y1=max(0,int(abs_y-abs_r+shrink))
                    x2=min(w_img,int(abs_x+abs_r-shrink))
                    y2=min(h_img,int(abs_y+abs_r-shrink))
                    crop = image[y1:y2, x1:x2]
                    if crop.size>0 and crop.shape[0]>50 and crop.shape[1]>50:
                        egg_crops.append(crop)

                if egg_crops:
                    print(f"[DEBUG] egg crops: {len(egg_crops)}")
                    return egg_crops
            else:
                print("[DEBUG] HoughCircles: tidak ada circle")
        else:
            print("[DEBUG] tray terlalu kecil, skip")

    print("[DEBUG] Fallback: 1 telur")
    return [image]


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Threshold dinamis
    try:
        thr_low  = float(request.form.get("threshold_low",  0.45))
        thr_high = float(request.form.get("threshold_high", 0.80))
    except (ValueError, TypeError):
        thr_low, thr_high = 0.45, 0.80

    thr_low  = max(0.01, min(thr_low,  0.99))
    thr_high = max(thr_low + 0.01, min(thr_high, 0.99))

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Gambar tidak bisa dibaca"}), 400

    print(f"IMAGE shape: {img.shape}")
    print(f"[THRESHOLD] low={thr_low:.2f}  high={thr_high:.2f}")

    eggs = detect_eggs_from_array(img)
    print(f"Telur terdeteksi: {len(eggs)}")

    results = []
    for i, egg in enumerate(eggs):

        # ── Display crop (BGR, untuk UI) ──────────────
        egg_display = cv2.resize(egg, (224, 224))
        _, buffer = cv2.imencode(
            '.jpg', egg_display,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        crop_b64 = base64.b64encode(buffer).decode('utf-8')

        # ── Preprocessing → model ─────────────────────
        processed = preprocess_image_from_array(egg)
        processed_float = processed.astype('float32') / 255.0

        input_tensor = np.expand_dims(processed_float, axis=0)
        prob_retak = float(model.predict(input_tensor, verbose=0)[0][0])
        prob_normal = 1 - prob_retak
        tingkat_kelayakan = prob_normal * 100

        # ── Zona klasifikasi ──────────────────────────
        if prob_retak < thr_low:
            status    = "Normal"
            zona      = "LAYAK"
            is_normal = True
        elif prob_retak < thr_high:
            status    = "Meragukan"
            zona      = "PERLU CEK"
            is_normal = True
        else:
            status    = "Retak"
            zona      = "TIDAK LAYAK"
            is_normal = False

        confidence = tingkat_kelayakan if is_normal else prob_retak * 100

        print(
            f"  Telur {i+1}: prob_retak={prob_retak:.3f} "
            f"→ {zona} ({tingkat_kelayakan:.1f}%)"
        )

        # ── GradCAM ───────────────────────────────────
        # Hanya generate kalau PERLU CEK atau TIDAK LAYAK
        # supaya tidak buang waktu untuk telur yang jelas LAYAK
        gradcam_b64 = None
        if zona in ("PERLU CEK", "TIDAK LAYAK"):
            gradcam_b64 = generate_gradcam(processed_float, egg_display.copy())
            if gradcam_b64:
                print(f"  [GradCAM] Telur {i+1}: OK")
            else:
                print(f"  [GradCAM] Telur {i+1}: gagal/skip")

        results.append({
            "egg":              i + 1,
            "status":           status,
            "zona":             zona,
            "confidence":       round(confidence, 2),
            "tingkat_kelayakan": round(tingkat_kelayakan, 2),
            "layak":            is_normal,
            "isLayak":          is_normal,
            "normalProb":       round(prob_normal * 100, 2),
            "retakProb":        round(prob_retak * 100, 2),
            "crop_image":       crop_b64,
            "gradcam_image":    gradcam_b64,   # None kalau LAYAK
        })

    layak       = sum(1 for r in results if r["zona"] == "LAYAK")
    perlu_cek   = sum(1 for r in results if r["zona"] == "PERLU CEK")
    tidak_layak = sum(1 for r in results if r["zona"] == "TIDAK LAYAK")

    return jsonify({
        "total":          len(results),
        "layak":          layak,
        "perlu_cek":      perlu_cek,
        "tidak_layak":    tidak_layak,
        "threshold_low":  round(thr_low,  2),
        "threshold_high": round(thr_high, 2),
        "results":        results
    })


# ─────────────────────────────────────────────
# ENDPOINT KHUSUS DETAIL CHECK (3 sudut)
# Dipanggil dari detail_check_screen.dart
# Menerima 1 foto, return predict + GradCAM
# ─────────────────────────────────────────────
@app.route("/predict_single", methods=["POST"])
def predict_single():
    """
    Endpoint ringan untuk mode detail check (single egg, 1 foto per call).
    Tidak ada deteksi nampan — langsung preprocess → predict → GradCAM.
    """
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Tidak ada gambar"}), 400

    try:
        thr_low  = float(request.form.get("threshold_low",  0.45))
        thr_high = float(request.form.get("threshold_high", 0.80))
    except (ValueError, TypeError):
        thr_low, thr_high = 0.45, 0.80

    thr_low  = max(0.01, min(thr_low,  0.99))
    thr_high = max(thr_low + 0.01, min(thr_high, 0.99))

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Gambar tidak bisa dibaca"}), 400

    # Display
    egg_display = cv2.resize(img, (224, 224))
    _, buffer = cv2.imencode('.jpg', egg_display, [cv2.IMWRITE_JPEG_QUALITY, 80])
    crop_b64 = base64.b64encode(buffer).decode('utf-8')

    # Preprocess
    processed = preprocess_image_from_array(img)
    processed_float = processed.astype('float32') / 255.0

    input_tensor = np.expand_dims(processed_float, axis=0)
    prob_retak = float(model.predict(input_tensor, verbose=0)[0][0])
    prob_normal = 1 - prob_retak
    tingkat_kelayakan = prob_normal * 100

    if prob_retak < thr_low:
        zona      = "LAYAK"
        is_normal = True
    elif prob_retak < thr_high:
        zona      = "PERLU CEK"
        is_normal = True
    else:
        zona      = "TIDAK LAYAK"
        is_normal = False

    confidence = tingkat_kelayakan if is_normal else prob_retak * 100

    # GradCAM selalu di-generate untuk detail check
    gradcam_b64 = generate_gradcam(processed_float, egg_display.copy())

    return jsonify({
        "zona":             zona,
        "prob_retak":       round(prob_retak * 100, 2),
        "prob_normal":      round(prob_normal * 100, 2),
        "tingkat_kelayakan": round(tingkat_kelayakan, 2),
        "confidence":       round(confidence, 2),
        "crop_image":       crop_b64,
        "gradcam_image":    gradcam_b64,
        "threshold_low":    round(thr_low,  2),
        "threshold_high":   round(thr_high, 2),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
