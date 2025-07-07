
import os
import cv2
import numpy as np
import pywt
import zlib
import base64
import hashlib
import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ========== Autoencoder for Denoising DWT Coefficients ==========
def build_autoencoder(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ========== Chen's Hyperchaotic Map ==========
def chen_hyperchaos(n, x0=[0.1, 0.2, 0.3, 0.4], dt=0.01):
    a, b, c, d = 35, 3, 28, 1
    x = np.array(x0)
    seq = []
    for _ in range(n):
        dx = np.array([
            a * (x[1] - x[0]),
            (c - a) * x[0] - x[0] * x[2] + c * x[1] - x[3],
            x[0] * x[1] - b * x[2],
            x[0] - d * x[3]
        ])
        x = x + dt * dx
        seq.append(x.copy())
    return np.array(seq)

# ========== DNA Mapping ==========
def binary_to_dna(bits):
    map_bin_to_dna = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
    return ''.join(map_bin_to_dna[bits[i:i+2]] for i in range(0, len(bits), 2))

def dna_to_binary(dna):
    map_dna_to_bin = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
    return ''.join(map_dna_to_bin[nuc] for nuc in dna)

# ========== QDE Embedding ==========
def qde_embed(coeffs, bits):
    coeffs_flat = coeffs.flatten()
    for i in range(0, len(bits)):
        x = coeffs_flat[i*2]
        y = coeffs_flat[i*2+1]
        l = (x + y) // 2
        h = x - y
        b = int(bits[i])
        h_prime = 2 * h + b
        x_prime = l + (h_prime + 1) // 2
        y_prime = l - h_prime // 2
        coeffs_flat[i*2] = x_prime
        coeffs_flat[i*2+1] = y_prime
    return coeffs_flat.reshape(coeffs.shape)

def qde_extract(coeffs, bit_len):
    coeffs_flat = coeffs.flatten()
    bits = []
    for i in range(bit_len):
        x = coeffs_flat[i*2]
        y = coeffs_flat[i*2+1]
        h_prime = x - y
        b = h_prime % 2
        bits.append(str(b))
    return ''.join(bits)

# ========== Payload Encryption ==========
def encrypt_payload(data_bytes, password):
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32)
    cipher = AES.new(key[:16], AES.MODE_EAX)
    compressed = zlib.compress(data_bytes)
    ciphertext, tag = cipher.encrypt_and_digest(compressed)
    return salt + cipher.nonce + tag + ciphertext

def decrypt_payload(payload, password):
    salt, nonce, tag, ciphertext = payload[:16], payload[16:32], payload[32:48], payload[48:]
    key = PBKDF2(password, salt, dkLen=32)
    cipher = AES.new(key[:16], AES.MODE_EAX, nonce=nonce)
    decompressed = cipher.decrypt_and_verify(ciphertext, tag)
    return zlib.decompress(decompressed)

# ========== Full Pipeline ==========
def embed_hybrid_dna_qde(input_image, secret_file, output_image, password):
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    autoencoder = build_autoencoder(LL.size)
    flat_LL = LL.flatten()
    autoencoder.fit(flat_LL[np.newaxis], flat_LL[np.newaxis], epochs=10, verbose=0)
    denoised = autoencoder.predict(flat_LL[np.newaxis])[0].reshape(LL.shape)

    with open(secret_file, 'rb') as f:
        raw = f.read()
    enc = encrypt_payload(raw, password)
    bits = ''.join(format(b, '08b') for b in enc)

    chaos_seq = chen_hyperchaos(len(bits) // 4 + 1)[:,0]
    mixed = ''.join(str((int(bits[i:i+4],2) ^ int((chaos_seq[i]*1e4)%16))) for i in range(0,len(bits),4))
    dna = binary_to_dna(mixed)
    bin_dna = ''.join(format(ord(c), '08b') for c in dna)

    LL_embed = qde_embed(denoised.copy(), bin_dna)
    coeffs_embed = (LL_embed, (LH, HL, HH))
    stego = pywt.idwt2(coeffs_embed, 'haar')
    cv2.imwrite(output_image, np.clip(stego,0,255).astype(np.uint8))

    print("✅ Secret embedded successfully.")


def extract_hybrid_dna_qde(stego_image, bit_len, password):
    img = cv2.imread(stego_image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')

    bin_dna = qde_extract(LL, bit_len)
    dna = ''.join(chr(int(bin_dna[i:i+8],2)) for i in range(0, len(bin_dna), 8))
    binary = dna_to_binary(dna)

    chaos_seq = chen_hyperchaos(len(binary)//4 + 1)[:,0]
    demix = ''.join(format(int(binary[i:i+1],2) ^ int((chaos_seq[i]*1e4)%16), '04b') for i in range(len(binary)))
    enc_bytes = bytearray(int(demix[i:i+8],2) for i in range(0,len(demix),8))

    data = decrypt_payload(enc_bytes, password)
    with open("recovered_secret", 'wb') as f:
        f.write(data)
    print("🔓 Secret recovered and saved.")
