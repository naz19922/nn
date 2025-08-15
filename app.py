import io, os
import cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, HTMLResponse
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from rembg import remove, new_session

# ---- Config ----
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/models/insightface_cache")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INSIGHTFACE_HOME, exist_ok=True)

INSWAPPER_PATH = os.path.join(MODELS_DIR, os.environ.get("INSWAPPER_ONNX", "inswapper_128.onnx"))

# ---- Helpers ----
def read_upload(u: UploadFile) -> np.ndarray:
    buf = np.frombuffer(u.file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Invalid image")
    return img

def to_jpeg_bytes(bgr: np.ndarray, quality=92) -> bytes:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bio = io.BytesIO()
    Image.fromarray(rgb).save(bio, format="JPEG", quality=quality)
    return bio.getvalue()

def ellipse_head_mask(shape, face):
    h,w = shape[:2]
    x1,y1,x2,y2 = map(int, face.bbox.astype(int))
    cx = (x1+x2)//2
    cy = int(y2 - 0.15*(y2-y1))
    ax = int(0.65*(x2-x1))
    ay = int(0.95*(y2-y1))
    m = np.zeros((h,w), np.uint8)
    cv2.ellipse(m, (cx,cy), (ax,ay), 0, 0, 360, 255, -1)
    return m

def dilate_feather(mask, dilate_px, feather_px):
    if dilate_px>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, k, 1)
    if feather_px>0:
        mask = cv2.GaussianBlur(mask, (feather_px*2+1, feather_px*2+1), 0)
    return mask

def color_transfer_reinhard(src_bgr, dst_bgr):
    def stats(c): return c.mean(), c.std()+1e-6
    sL = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        ms, ss = stats(sL[...,i]); md, sd = stats(dL[...,i])
        dL[...,i] = ((dL[...,i]-md)*(ss/sd))+ms
    dL = np.clip(dL,0,255).astype(np.uint8)
    return cv2.cvtColor(dL, cv2.COLOR_LAB2BGR)

# ---- App & Models ----
app = FastAPI(title="HeadSwap API", version="1.1")

face_app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_HOME)
face_app.prepare(ctx_id=0, det_size=(640,640))

try:
    if os.path.exists(INSWAPPER_PATH):
        inswapper = get_model(INSWAPPER_PATH, download=False, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    else:
        inswapper = get_model("inswapper_128.onnx", download=True, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
except Exception as e:
    inswapper = None
    print("InSwapper load error:", e)

rembg_session = new_session("u2net_human_seg")  # auto-downloads

def head_mask_rembg(img_bgr: np.ndarray, face) -> np.ndarray:
    # person alpha from rembg, intersect with ellipse around face => head/neck mask
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bio = io.BytesIO(); Image.fromarray(img_rgb).save(bio, format="PNG")
    alpha_png = remove(bio.getvalue(), session=rembg_session)
    alpha = np.array(Image.open(io.BytesIO(alpha_png)).convert("RGBA"))[...,3]
    ell = ellipse_head_mask(img_bgr.shape, face)
    return cv2.bitwise_and(alpha, ell)

# ---- Routes ----
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><body>
    <h2>Head Swap (face + hair)</h2>
    <form method="post" action="/swap_ui" enctype="multipart/form-data">
      <p>Source face: <input type="file" name="source" required></p>
      <p>Target image: <input type="file" name="target" required></p>
      <p>Dilation(px): <input type="number" name="dilation" value="18"></p>
      <p>Feather(px): <input type="number" name="feather" value="14"></p>
      <p>JPEG quality: <input type="number" name="jpeg_quality" value="92"></p>
      <button type="submit">Swap</button>
    </form>
    </body></html>
    """

@app.post("/swap_ui")
def swap_ui(source: UploadFile = File(...), target: UploadFile = File(...),
           dilation: int = Form(18), feather: int = Form(14), jpeg_quality: int = Form(92)):
    res = swap(source, target, dilation=dilation, feather=feather, do_color_match=True, jpeg_quality=jpeg_quality)
    if isinstance(res, JSONResponse): return res
    return Response(content=res.body, media_type="image/jpeg")

@app.post("/swap")
def swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    dilation: int = Form(18),
    feather: int = Form(14),
    do_color_match: bool = Form(True),
    jpeg_quality: int = Form(92),
):
    try:
        if inswapper is None:
            return JSONResponse({"error":"InSwapper model not loaded."}, status_code=500)

        src = read_upload(source)
        dst = read_upload(target)

        src_faces = face_app.get(src)
        dst_faces = face_app.get(dst)
        if not src_faces: return JSONResponse({"error":"No face in source"}, status_code=400)
        if not dst_faces: return JSONResponse({"error":"No face in target"}, status_code=400)

        s = max(src_faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        d = max(dst_faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        swapped_crop = inswapper.get(dst, d, s, paste_back=False)

        mask = head_mask_rembg(dst, d)
        mask = dilate_feather(mask, dilation, feather)

        x1,y1,x2,y2 = map(int, d.bbox.astype(int))
        ch, cw = y2-y1, x2-x1
        swapped_resized = cv2.resize(swapped_crop, (cw, ch), interpolation=cv2.INTER_LINEAR)

        if do_color_match:
            yA,yB = max(0,y1-10), min(dst.shape[0], y2+10)
            xA,xB = max(0,x1-10), min(dst.shape[1], x2+10)
            head_dst = dst[yA:yB, xA:xB]
            if head_dst.size>0:
                swapped_resized = color_transfer_reinhard(head_dst, swapped_resized)

        src_canvas = np.zeros_like(dst)
        src_canvas[y1:y2, x1:x2] = swapped_resized

        mask3 = cv2.merge([mask,mask,mask])
        ys, xs = np.nonzero(mask)
        center = ((x1+x2)//2, (y1+y2)//2) if len(xs)==0 else (int(xs.mean()), int(ys.mean()))
        blended = cv2.seamlessClone(src_canvas, dst, mask3, center, cv2.NORMAL_CLONE)

        k = np.array([[0,-1,0],[-1,5.0,-1],[0,-1,0]], np.float32)
        blended = cv2.filter2D(blended, -1, k)

        return Response(content=to_jpeg_bytes(blended, quality=int(jpeg_quality)), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
