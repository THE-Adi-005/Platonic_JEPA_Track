"""
APAD: Attribute-Stratified Platonic Alignment Distance & Singular Value Spectrum
Computes geometric cross-modal alignment based on Balestriero & LeCun (arXiv:2610.39302).
"""

import numpy as np
import scipy.linalg


def compute_procrustes_rotation(Z_v: np.ndarray, Z_a: np.ndarray) -> tuple:
    """
    Solves the Orthogonal Procrustes problem:
    R* = argmin_{R in O(d)} || Z_v R - Z_a ||_F^2
    Via SVD: Z_v^T Z_a = U Sigma V^T  ==> R* = U V^T
    
    Z_v: (N, d) video embeddings
    Z_a: (N, d) attribute embeddings
    returns: (R, singular_values, residual_error)
    """
    # Cross-covariance matrix
    M = Z_v.T @ Z_a # (d, d)
    U, S, Vt = scipy.linalg.svd(M)
    R = U @ Vt

    # Residual alignment error normalized by sqrt(d)
    diff = Z_v @ R - Z_a
    residual_error = np.linalg.norm(diff, ord='fro') / (np.sqrt(Z_v.shape[1] * Z_v.shape[0]))

    return R, S, residual_error


def compute_singular_value_spectrum(Z_v: np.ndarray, Z_a: np.ndarray) -> np.ndarray:
    """
    Computes normalized singular value spectrum: S / S[0].
    Under Good Views, spectrum is flat/uniform.
    Under Bad Views, spectrum decays exponentially.
    """
    M = Z_v.T @ Z_a
    _, S, _ = scipy.linalg.svd(M)
    if len(S) > 0 and S[0] > 1e-12:
        return S / S[0]
    return S


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between representations X and Y.
    Returns scalar in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    dot_prod = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    
    if norm_x * norm_y < 1e-12:
        return 0.0
    return float(dot_prod / (norm_x * norm_y))


def compute_stratified_apad(
    Z_v_dict: dict,
    Z_a_dict: dict
) -> dict:
    """
    Computes Attribute-Stratified Platonic Alignment Distance (APAD) per attribute:
    Gender, Age, Emotion, Profession, Style.
    
    Returns dict mapping attribute name -> {apad_distance, cka, singular_spectrum}
    """
    results = {}
    for attr_name in Z_v_dict.keys():
        Z_v = Z_v_dict[attr_name]
        Z_a = Z_a_dict[attr_name]
        
        R, S, res_err = compute_procrustes_rotation(Z_v, Z_a)
        cka = compute_linear_cka(Z_v, Z_a)
        norm_s = S / S[0] if len(S) > 0 and S[0] > 0 else S

        results[attr_name] = {
            "apad_distance": float(res_err),
            "cka_alignment": float(cka),
            "top_5_singular_values": [float(x) for x in norm_s[:5]]
        }
    return results
