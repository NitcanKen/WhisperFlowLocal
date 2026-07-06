"""Generate AppIcon.icns from a source image: center-crop to square,
apply the macOS rounded-square mask, render all iconset sizes.

Usage: .venv/bin/python scripts/make_icon.py <source-image> [out.icns]
"""
import os
import subprocess
import sys
import tempfile

import objc  # noqa: F401  (ensures pyobjc initialized first)
from AppKit import (
    NSBezierPath,
    NSBitmapImageRep,
    NSCompositingOperationCopy,
    NSGraphicsContext,
    NSImage,
    NSMakeRect,
    NSPNGFileType,
)

# Apple's Big-Sur-style icon grid: content occupies ~80% of the canvas,
# corner radius ~22.37% of the content size.
CANVAS = 1024
CONTENT = 824
RADIUS = CONTENT * 0.2237

SIZES = [16, 32, 64, 128, 256, 512, 1024]


def render_master(src_path: str, out_png: str) -> None:
    src = NSImage.alloc().initWithContentsOfFile_(src_path)
    if src is None:
        raise SystemExit(f"cannot read image: {src_path}")
    sw, sh = src.size().width, src.size().height
    side = min(sw, sh)
    crop = NSMakeRect((sw - side) / 2.0, (sh - side) / 2.0, side, side)

    rep = NSBitmapImageRep.alloc(
    ).initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, CANVAS, CANVAS, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0
    )
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)

    inset = (CANVAS - CONTENT) / 2.0
    target = NSMakeRect(inset, inset, CONTENT, CONTENT)
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        target, RADIUS, RADIUS
    ).addClip()
    src.drawInRect_fromRect_operation_fraction_(
        target, crop, NSCompositingOperationCopy, 1.0
    )

    NSGraphicsContext.restoreGraphicsState()
    png = rep.representationUsingType_properties_(NSPNGFileType, None)
    png.writeToFile_atomically_(out_png, True)


def main() -> int:
    src = sys.argv[1]
    out_icns = sys.argv[2] if len(sys.argv) > 2 else "AppIcon.icns"
    tmp = tempfile.mkdtemp(prefix="wfl_icon_")
    master = os.path.join(tmp, "master.png")
    render_master(src, master)

    iconset = os.path.join(tmp, "AppIcon.iconset")
    os.makedirs(iconset)
    for s in SIZES:
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = s * scale
            if px > 1024:
                continue
            name = f"icon_{s}x{s}{suffix}.png"
            subprocess.run(
                ["sips", "-z", str(px), str(px), master, "--out",
                 os.path.join(iconset, name)],
                check=True, capture_output=True,
            )
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out_icns],
                   check=True)
    print(f"icon written: {out_icns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
