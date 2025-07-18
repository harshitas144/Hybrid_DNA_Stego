import os
import cv2
import numpy as np
import zlib
import hashlib
import struct
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
import gc
import warnings
import random
warnings.filterwarnings('ignore')

def get_file_extension(filename):
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()

def detect_file_type(data):
    """Detect file type from binary data"""
    # Common file signatures
    signatures = {
        b'%PDF-': '.pdf',
        b'\x89PNG\r\n\x1a\n': '.png',
        b'\xff\xd8\xff': '.jpg',
        b'GIF87a': '.gif',
        b'GIF89a': '.gif',
        b'PK\x03\x04': '.zip',
        b'PK\x05\x06': '.zip',
        b'PK\x07\x08': '.zip',
        b'\x50\x4b\x03\x04': '.docx',  # Also zip-based
        b'BM': '.bmp',
        b'RIFF': '.wav',
        b'\x00\x00\x01\x00': '.ico',
        b'\x7fELF': '.elf',
        b'MZ': '.exe',
    }

    for sig, ext in signatures.items():
        if data.startswith(sig):
            return ext

    # Check if it's plain text
    try:
        data.decode('utf-8')
        return '.txt'
    except UnicodeDecodeError:
        pass

    return '.bin'  # Unknown binary file

# ========== DNA Encoding/Decoding Functions ==========
def binary_to_dna(binary_str):
    """Convert binary string to DNA sequence using A, T, G, C"""
    # Mapping: 00->A, 01->T, 10->G, 11->C
    dna_map = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
    
    # Pad binary string to make it divisible by 2
    if len(binary_str) % 2 != 0:
        binary_str += '0'
    
    dna_sequence = ''
    for i in range(0, len(binary_str), 2):
        pair = binary_str[i:i+2]
        dna_sequence += dna_map[pair]
    
    return dna_sequence

def dna_to_binary(dna_sequence):
    """Convert DNA sequence back to binary string"""
    # Reverse mapping: A->00, T->01, G->10, C->11
    binary_map = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
    
    binary_str = ''
    for nucleotide in dna_sequence:
        if nucleotide in binary_map:
            binary_str += binary_map[nucleotide]
        else:
            # Handle invalid characters by treating as 'A'
            binary_str += '00'
    
    return binary_str

def generate_dna_key(password, length):
    """Generate a DNA key sequence from password for XOR encryption"""
    # Create deterministic random sequence from password
    seed = hashlib.sha256(password.encode()).hexdigest()
    random.seed(seed)
    
    nucleotides = ['A', 'T', 'G', 'C']
    dna_key = ''.join(random.choices(nucleotides, k=length))
    
    return dna_key

def xor_dna_sequences(seq1, seq2):
    """XOR two DNA sequences by converting to binary"""
    if len(seq1) != len(seq2):
        # Pad shorter sequence
        max_len = max(len(seq1), len(seq2))
        seq1 = seq1.ljust(max_len, 'A')
        seq2 = seq2.ljust(max_len, 'A')
    
    bin1 = dna_to_binary(seq1)
    bin2 = dna_to_binary(seq2)
    
    # XOR the binary representations
    result_bin = ''
    for i in range(len(bin1)):
        bit1 = int(bin1[i])
        bit2 = int(bin2[i])
        result_bin += str(bit1 ^ bit2)
    
    return binary_to_dna(result_bin)

def embed_dna_in_image(img, dna_sequence):
    """Embed DNA sequence in image using nucleotide-to-pixel mapping"""
    img_flat = img.flatten()
    
    if len(dna_sequence) > len(img_flat):
        raise ValueError(f"DNA sequence too long: {len(dna_sequence)} nucleotides, image capacity: {len(img_flat)}")
    
    # Create mapping: A=0, T=1, G=2, C=3 (using 2 LSBs)
    nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    
    img_modified = img_flat.copy().astype(np.int32)
    
    for i, nucleotide in enumerate(dna_sequence):
        pixel_val = img_modified[i]
        nucleotide_val = nucleotide_map[nucleotide]
        
        # Clear 2 LSBs and set new value
        img_modified[i] = (pixel_val & 0xFC) | nucleotide_val
    
    return img_modified.reshape(img.shape).astype(np.uint8)

def extract_dna_from_image(img, sequence_length):
    """Extract DNA sequence from image"""
    img_flat = img.flatten()
    
    if sequence_length > len(img_flat):
        raise ValueError(f"Sequence too long: {sequence_length} nucleotides, image size: {len(img_flat)}")
    
    # Reverse mapping: 0=A, 1=T, 2=G, 3=C
    value_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    
    dna_sequence = ''
    for i in range(sequence_length):
        pixel_val = int(img_flat[i])
        nucleotide_val = pixel_val & 0x03  # Extract 2 LSBs
        dna_sequence += value_map[nucleotide_val]
    
    return dna_sequence

# ========== DNA-based Hamming Error Correction ==========
def add_dna_hamming_correction(dna_sequence):
    """Add Hamming error correction to DNA sequence"""
    # Convert to binary, apply standard Hamming, then back to DNA
    binary_data = dna_to_binary(dna_sequence)
    corrected_binary = add_hamming_error_correction(binary_data)
    return binary_to_dna(corrected_binary)

def correct_dna_hamming_errors(dna_sequence):
    """Correct errors in DNA sequence using Hamming codes"""
    # Convert to binary, apply correction, then back to DNA
    binary_data = dna_to_binary(dna_sequence)
    corrected_binary, error_count = correct_hamming_errors(binary_data)
    return binary_to_dna(corrected_binary), error_count

# ========== Standard Hamming Functions (kept for DNA use) ==========
def add_hamming_error_correction(data_bits):
    """Hamming(7,4) error correction"""
    if len(data_bits) == 0:
        return ""

    result = []

    # Process in 4-bit blocks
    for i in range(0, len(data_bits), 4):
        block = data_bits[i:i+4]

        # Pad block to 4 bits if necessary
        while len(block) < 4:
            block += '0'

        # Convert to integers
        d = [int(b) for b in block]

        # Calculate parity bits for Hamming(7,4)
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]

        # Create 7-bit codeword: p1 p2 d0 p3 d1 d2 d3
        codeword = [p1, p2, d[0], p3, d[1], d[2], d[3]]
        result.extend([str(b) for b in codeword])

    return ''.join(result)

def correct_hamming_errors(coded_bits):
    """Correct Hamming errors"""
    if len(coded_bits) == 0:
        return "", 0

    corrected = []
    error_count = 0

    # Process only complete 7-bit blocks
    num_blocks = len(coded_bits) // 7
    print(f"📊 Processing {num_blocks} Hamming blocks from {len(coded_bits)} bits")

    for i in range(num_blocks):
        start = i * 7
        block = coded_bits[start:start+7]

        if len(block) < 7:
            break

        try:
            r = [int(b) for b in block]

            # Calculate syndrome
            s1 = r[0] ^ r[2] ^ r[4] ^ r[6]
            s2 = r[1] ^ r[2] ^ r[5] ^ r[6]
            s3 = r[3] ^ r[4] ^ r[5] ^ r[6]

            error_pos = s1 + 2*s2 + 4*s3

            # Correct single-bit error
            if error_pos != 0 and error_pos <= 7:
                r[error_pos - 1] ^= 1
                error_count += 1

            # Extract data bits: positions 2, 4, 5, 6 (0-indexed)
            data_bits = [r[2], r[4], r[5], r[6]]
            corrected.extend([str(b) for b in data_bits])

        except (ValueError, IndexError) as e:
            print(f"⚠️  Error processing block {i}: {e}")
            corrected.extend(['0', '0', '0', '0'])

    return ''.join(corrected), error_count

# ========== Encryption/Decryption Functions ==========
def encrypt_data_robust(data_bytes, password):
    """Encrypt data with compression and integrity checking"""
    if len(data_bytes) == 0:
        raise ValueError("Cannot encrypt empty data")

    # Add checksum for integrity
    checksum = hashlib.sha256(data_bytes).digest()
    data_with_checksum = data_bytes + checksum

    # Compress
    try:
        compressed = zlib.compress(data_with_checksum, level=6)
    except Exception as e:
        raise ValueError(f"Compression failed: {e}")

    # Create header with magic number and sizes
    magic = 0x12345678
    original_size = len(data_bytes)
    compressed_size = len(compressed)

    header = struct.pack('<III', magic, original_size, compressed_size)

    # Generate encryption key
    salt = hashlib.sha256(b'dna_steg_salt_2024').digest()[:16]
    key = PBKDF2(password, salt, dkLen=32, count=100000)

    # Encrypt using AES-GCM
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(header + compressed)

    # Combine: nonce(16) + tag(16) + ciphertext
    payload = cipher.nonce + tag + ciphertext

    print(f"📊 Encryption: {len(data_bytes)} -> {len(compressed)} -> {len(payload)} bytes")

    return payload

def decrypt_data_robust(payload, password):
    """Decrypt data with integrity verification"""
    try:
        print(f"🔍 Starting decryption of {len(payload)} bytes")
        
        if len(payload) < 32:
            raise ValueError(f"Payload too short: {len(payload)} bytes")

        # Extract components
        nonce = payload[:16]
        tag = payload[16:32]
        ciphertext = payload[32:]

        # Generate decryption key
        salt = hashlib.sha256(b'dna_steg_salt_2024').digest()[:16]
        key = PBKDF2(password, salt, dkLen=32, count=100000)

        # Decrypt
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        
        try:
            decrypted = cipher.decrypt_and_verify(ciphertext, tag)
        except ValueError as e:
            print(f"❌ AES-GCM verification failed: {e}")
            # Try fallback without verification
            fallback_cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            decrypted = fallback_cipher.decrypt(ciphertext)
            print(f"⚠️  Decrypted without tag verification")

        # Extract header
        if len(decrypted) < 12:
            raise ValueError(f"Decrypted data too short for header")

        header = decrypted[:12]
        compressed_data = decrypted[12:]

        # Unpack header
        magic, original_size, compressed_size = struct.unpack('<III', header)
        
        if magic != 0x12345678:
            raise ValueError(f"Invalid magic number: 0x{magic:08x}")

        # Decompress
        data_with_checksum = zlib.decompress(compressed_data)

        # Extract data and verify checksum
        data_bytes = data_with_checksum[:-32]
        expected_checksum = data_with_checksum[-32:]

        actual_checksum = hashlib.sha256(data_bytes).digest()
        if actual_checksum != expected_checksum:
            print(f"⚠️  Checksum verification failed")

        return data_bytes

    except Exception as e:
        print(f"❌ Decryption error: {e}")
        raise

# ========== Main DNA Steganography Functions ==========
def embed_dna_steganography(input_image, secret_file, output_image, password):
    """Embed secret data in image using DNA sequence encoding"""
    print("🧬 Starting DNA-based steganography embedding...")
    
    # Load image
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {input_image}")

    print(f"📊 Image loaded: {img.shape}")

    # Load and encrypt secret
    print("📊 Loading and encrypting secret...")
    with open(secret_file, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) == 0:
        raise ValueError("Secret file is empty")

    encrypted_data = encrypt_data_robust(raw_data, password)

    # Convert to binary and then to DNA
    data_bits = ''.join(format(b, '08b') for b in encrypted_data)
    print(f"📊 Data bits: {len(data_bits)}")

    # Add Hamming error correction
    print("📊 Adding Hamming error correction...")
    corrected_bits = add_hamming_error_correction(data_bits)
    
    # Convert to DNA sequence
    dna_sequence = binary_to_dna(corrected_bits)
    print(f"🧬 DNA sequence length: {len(dna_sequence)} nucleotides")

    # Generate DNA key for additional security
    dna_key = generate_dna_key(password, len(dna_sequence))
    print(f"🔑 Generated DNA key: {len(dna_key)} nucleotides")

    # XOR with DNA key
    encrypted_dna = xor_dna_sequences(dna_sequence, dna_key)
    print(f"🧬 Encrypted DNA sequence: {len(encrypted_dna)} nucleotides")

    # Check capacity
    if len(encrypted_dna) > img.size:
        raise ValueError(f"DNA sequence too long: {len(encrypted_dna)} nucleotides, image capacity: {img.size}")

    # Embed DNA in image
    print("📊 Embedding DNA sequence in image...")
    stego_img = embed_dna_in_image(img, encrypted_dna)

    # Save as PNG to preserve quality
    output_png = output_image.replace('.jpg', '.png')
    cv2.imwrite(output_png, stego_img)
    print(f"✅ DNA stego image saved: {output_png}")

    # Cleanup
    del img, stego_img
    gc.collect()

    # Return metadata
    metadata = {
        'data_bits_length': len(data_bits),
        'corrected_bits_length': len(corrected_bits),
        'dna_sequence_length': len(encrypted_dna),
        'original_payload_length': len(encrypted_data),
        'method': 'dna',
        'error_correction': 'hamming',
        'image_file': output_png
    }

    return metadata

def extract_dna_steganography(stego_image, metadata, password, output_filename="recovered_secret"):
    """Extract secret data from DNA-encoded steganography"""
    print("🧬 Starting DNA-based steganography extraction...")
    
    # Load stego image
    img = cv2.imread(stego_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {stego_image}")

    print(f"📊 Image loaded: {img.shape}")

    dna_sequence_length = metadata['dna_sequence_length']
    print(f"🧬 Expected DNA sequence length: {dna_sequence_length}")

    # Extract DNA sequence from image
    print("📊 Extracting DNA sequence...")
    encrypted_dna = extract_dna_from_image(img, dna_sequence_length)
    print(f"🧬 Extracted DNA sequence: {len(encrypted_dna)} nucleotides")

    # Generate the same DNA key
    dna_key = generate_dna_key(password, len(encrypted_dna))
    print(f"🔑 Generated DNA key: {len(dna_key)} nucleotides")

    # Decrypt DNA sequence
    dna_sequence = xor_dna_sequences(encrypted_dna, dna_key)
    print(f"🧬 Decrypted DNA sequence: {len(dna_sequence)} nucleotides")

    # Convert DNA back to binary
    corrected_bits = dna_to_binary(dna_sequence)
    print(f"📊 Converted to binary: {len(corrected_bits)} bits")

    # Apply Hamming error correction
    print("📊 Applying Hamming error correction...")
    corrected_data, error_count = correct_hamming_errors(corrected_bits)
    total_blocks = len(corrected_bits) // 7
    error_rate = error_count / total_blocks if total_blocks > 0 else 0
    print(f"📊 Error correction: {error_count} errors, rate: {error_rate:.4f}")

    # Handle data length
    expected_length = metadata['data_bits_length']
    if len(corrected_data) > expected_length:
        corrected_data = corrected_data[:expected_length]
    elif len(corrected_data) < expected_length:
        corrected_data = corrected_data.ljust(expected_length, '0')

    # Convert to bytes
    try:
        encrypted_bytes = bytearray()
        for i in range(0, len(corrected_data), 8):
            byte_bits = corrected_data[i:i+8]
            if len(byte_bits) == 8:
                encrypted_bytes.append(int(byte_bits, 2))
        
        encrypted_bytes = bytes(encrypted_bytes)
        
    except ValueError as e:
        raise ValueError(f"Error converting bits to bytes: {e}")

    print(f"📊 Reconstructed payload: {len(encrypted_bytes)} bytes")

    # Decrypt
    print("📊 Decrypting...")
    try:
        decrypted_data = decrypt_data_robust(encrypted_bytes, password)

        # Auto-detect file type
        detected_extension = detect_file_type(decrypted_data)
        print(f"🔍 Detected file type: {detected_extension}")

        # Save recovered file
        base, _ = os.path.splitext(output_filename)
        output_path = base + detected_extension

        with open(output_path, 'wb') as f:
            f.write(decrypted_data)

        print(f"✅ Secret recovered using DNA steganography: {output_path}")
        print(f"📊 Recovered {len(decrypted_data)} bytes")

        # Cleanup
        del img
        gc.collect()

        return True, output_path

    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        return False

# ========== Helper Functions ==========
def calculate_dna_capacity(image_path):
    """Calculate DNA steganography capacity"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"📏 Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Each pixel can store 1 DNA nucleotide (using 2 LSBs)
    # Each nucleotide represents 2 bits
    raw_capacity_nucleotides = img.size
    raw_capacity_bits = raw_capacity_nucleotides * 2
    
    print(f"📏 Raw capacity: {raw_capacity_nucleotides} nucleotides ({raw_capacity_bits} bits)")

    # Account for Hamming(7,4) code: 4 data bits per 7 coded bits
    effective_capacity_bits = (raw_capacity_bits // 7) * 4
    effective_capacity_bytes = effective_capacity_bits // 8
    
    print(f"📏 Effective capacity: {effective_capacity_bits} bits ({effective_capacity_bytes} bytes)")

    return effective_capacity_bytes

def check_dna_compatibility(image_path, secret_file):
    """Check if secret file can be embedded using DNA steganography"""
    try:
        capacity = calculate_dna_capacity(image_path)

        with open(secret_file, 'rb') as f:
            data = f.read()

        # Estimate overhead: encryption + compression + checksum
        estimated_size = len(data) + 100  # rough overhead estimate
        
        print(f"📦 Original file size: {len(data)} bytes")
        print(f"📦 Estimated size needed: {estimated_size} bytes")

        if estimated_size <= capacity:
            print(f"✅ Compatible! Spare capacity: {capacity - estimated_size} bytes")
            return True
        else:
            print(f"❌ Not compatible! Shortfall: {estimated_size - capacity} bytes")
            return False

    except Exception as e:
        print(f"❌ Error checking compatibility: {e}")
        return False
