"""5-point similarity-transform face alignment (P2-2).

Alignment is what makes a frozen visual backbone usable: without it the network
spends capacity modelling head pose instead of the artefacts we care about. The
transform is *similarity* (rotation + uniform scale + translation) and deliberately
not affine — shear would distort facial geometry, and geometry is signal here.

The solver is Umeyama's closed form rather than `cv2.estimateAffinePartial2D`, for
one reason: `estimateAffinePartial2D` defaults to RANSAC, which samples randomly and
is therefore not bit-reproducible across runs. P2-11 requires byte-identical output,
so every step of this path is analytic.

The destination template is the ArcFace 112x112 5-point layout, the de-facto standard
for face-recognition-style crops, scaled by `margin` to include surrounding context
(PROJECT_PLAN section B: `margin=0.25`).
"""

from __future__ import annotations

import numpy as np

# ArcFace canonical 5-point template for a 112x112 crop, in (x, y) image coordinates.
# Order: left eye, right eye, nose tip, left mouth corner, right mouth corner --
# "left" meaning image-left (the viewer's left), not the subject's left.
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def template(crop_size: int = 112, margin: float = 0.25) -> np.ndarray:
    """The destination 5 points for a `crop_size` square, widened by `margin`.

    `margin` zooms *out*: the canonical face is shrunk toward the crop centre so the
    crop includes jaw, hairline and some background. margin=0 reproduces ArcFace
    exactly; margin=0.25 scales the face to 1/1.25 = 80% of the frame.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    pts = ARCFACE_112 * (crop_size / 112.0)
    centre = np.array([crop_size / 2.0, crop_size / 2.0])
    return centre + (pts - centre) / (1.0 + margin)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping `src` onto `dst`.

    Umeyama (1991), "Least-squares estimation of transformation parameters between
    two point patterns". Returns a 2x3 matrix suitable for `cv2.warpAffine`.

    Fully deterministic: no sampling, no iteration, no RNG.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"expected matching (N, 2) arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 correspondences, got {n}")

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    # Cross-covariance, then its SVD.
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)

    # Reflection guard: a similarity transform may rotate but must never mirror.
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0

    rot = u @ np.diag(d) @ vt

    src_var = (src_c**2).sum() / n
    scale = 1.0 if src_var < 1e-12 else (s * d).sum() / src_var

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rot
    matrix[:, 2] = dst_mean - scale * rot @ src_mean
    return matrix


def align_face(
    frame: np.ndarray,
    landmarks5: np.ndarray,
    *,
    crop_size: int = 112,
    margin: float = 0.25,
) -> np.ndarray:
    """Warp `frame` so the 5 landmarks land on the canonical template.

    Returns a `(crop_size, crop_size, 3)` uint8 array. Border pixels are replicated
    rather than zero-filled: a black wedge at the frame edge is a hard edge the
    backbone will happily learn as a feature, which is worse than a smeared one.
    """
    import cv2

    matrix = umeyama(np.asarray(landmarks5, dtype=np.float64), template(crop_size, margin))
    return cv2.warpAffine(
        frame,
        matrix.astype(np.float32),
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
