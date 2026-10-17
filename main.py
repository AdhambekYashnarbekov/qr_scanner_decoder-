
import io
import logging
from urllib.parse import urlencode

import cv2
import numpy as np
import qrcode
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import ImageColor, UnidentifiedImageError
from qrcode.exceptions import DataOverflowError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QR Code Studio")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_SIZES = [5, 8, 10, 12, 15, 20, 30, 40]


def valid_color(color: str) -> bool:
    """Return True if color is a valid Pillow color."""
    try:
        ImageColor.getrgb(color)
        return True
    except (ValueError, TypeError):
        return False


def render_page(request: Request, **context):
    """Render the main page with default values."""
    defaults = {
        "data": "",
        "size": 10,
        "dark_color": "#000000",
        "light_color": "#FFFFFF",
        "image_url": None,
        "error": None,
        "decoded_data": None,
        "decode_error": None,
    }
    defaults.update(context)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=defaults,
    )


@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    data: str = "",
    size: int = 10,
    dark_color: str = "#000000",
    light_color: str = "#FFFFFF",
):
    data = data.strip()
    error = None
    image_url = None

    if data:
        if len(data) > 5000:
            error = "Please enter no more than 5,000 characters."
        elif size not in ALLOWED_SIZES:
            error = "Please select a valid QR code size."
        elif not valid_color(dark_color):
            error = "The dark color is invalid."
        elif not valid_color(light_color):
            error = "The light color is invalid."
        else:
            params = urlencode({
                "data": data,
                "size": size,
                "dark_color": dark_color,
                "light_color": light_color,
            })
            image_url = f"/generate?{params}"

    return render_page(
        request,
        data=data,
        size=size if size in ALLOWED_SIZES else 10,
        dark_color=dark_color if valid_color(dark_color) else "#000000",
        light_color=light_color if valid_color(light_color) else "#FFFFFF",
        image_url=image_url,
        error=error,
    )


@app.get("/generate")
async def generate_qr(
    data: str,
    size: int = 10,
    dark_color: str = "#000000",
    light_color: str = "#FFFFFF",
    download: int = 0,
):
    data = data.strip()

    if not data:
        return Response("Please provide data for the QR code.", status_code=400)

    if len(data) > 5000:
        return Response("Input is too long.", status_code=400)

    if size not in ALLOWED_SIZES:
        return Response("Invalid QR code size.", status_code=400)

    if not valid_color(dark_color) or not valid_color(light_color):
        return Response("Invalid color.", status_code=400)

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=size,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        image = qr.make_image(
            fill_color=dark_color,
            back_color=light_color,
        ).convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        headers = {}
        if download == 1:
            headers["Content-Disposition"] = (
                'attachment; filename="qrcode.png"'
            )

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers=headers,
        )

    except DataOverflowError:
        return Response(
            "The provided data is too large for a QR code.",
            status_code=400,
        )
    except Exception:
        logger.exception("QR code generation failed")
        return Response(
            "Unable to generate the QR code.",
            status_code=500,
        )


@app.post("/decode", response_class=HTMLResponse)
async def decode_qr(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        if file.content_type and not file.content_type.startswith("image/"):
            return render_page(
                request,
                decode_error="Please upload an image file.",
            )

        contents = await file.read(MAX_FILE_SIZE + 1)

        if not contents:
            return render_page(
                request,
                decode_error="The uploaded file is empty.",
            )

        if len(contents) > MAX_FILE_SIZE:
            return render_page(
                request,
                decode_error="The image must be smaller than 10 MB.",
            )

        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return render_page(
                request,
                decode_error="Could not read the image. Try a PNG or JPEG.",
            )

        detector = cv2.QRCodeDetector()
        decoded_data, points, _ = detector.detectAndDecode(image)

        if not decoded_data:
            return render_page(
                request,
                decode_error=(
                    "No readable QR code was found. "
                    "Try a clearer image with the entire QR code visible."
                ),
            )

        return render_page(request, decoded_data=decoded_data)

    except Exception:
        logger.exception("QR code decoding failed")
        return render_page(
            request,
            decode_error="An unexpected error occurred while decoding.",
        )
    finally:
        await file.close()


@app.get("/health")
async def health():
    return {"status": "ok"}
