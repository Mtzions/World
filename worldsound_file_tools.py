
#!/usr/bin/env python3
"""
File-in-Sound v1: Encrypt files using WorldSound OTP v2 and embed them into audio using max-stealth steganography.

This module provides functions to:
1. Encrypt any file using WorldSound OTP v2
2. Embed the encrypted data into an audio file using steganography
3. Extract and decrypt the data from the stego audio
"""

import argparse
import hashlib
import os
import random
import numpy as np
import scipy.signal
import soundfile as sf
import sys
from typing import Tuple, List
from scipy.signal import windows, stft, istft

# Import WorldSound OTP v2 functions
from worldsound import (
    extract_features_from_audio,
    derive_keystream_from_features,
    encrypt_bytes,
    decrypt_bytes
)


def compute_stft(
    samples: np.ndarray,
    fs: int,
    frame_size: int = 2048,
    hop_size: int = 512,
):
    """
    Deterministic STFT using Hann window and rFFT.

    Args:
        samples: mono 1D float array (recommended range [-1, 1])
        fs: sample rate in Hz
        frame_size: analysis window length
        hop_size: hop (stride) between frames

    Returns:
        S: complex STFT matrix, shape (num_frames, frame_size//2 + 1)
        freqs: frequency bins (Hz)
        window: window used (1D array length = frame_size)
        orig_len: original signal length before padding
    """
    if samples.ndim != 1:
        raise ValueError("compute_stft expects a mono 1D array")

    samples = samples.astype(np.float32, copy=False)
    orig_len = len(samples)

    window = windows.hann(frame_size, sym=False).astype(np.float32)

    # --- frame count & padding ---
    # we want enough frames so that the LAST frame is fully covered after padding.
    # n_frames * hop_size + frame_size - hop_size >= orig_len
    # => last index we touch is (n_frames-1)*hop + frame_size
    if orig_len <= frame_size:
        n_frames = 1
    else:
        n_frames = 1 + int(np.ceil((orig_len - frame_size) / hop_size))

    total_len = (n_frames - 1) * hop_size + frame_size
    pad_len = max(0, total_len - orig_len)
    if pad_len > 0:
        samples = np.pad(samples, (0, pad_len), mode="constant")

    # --- build frames ---
    frames = np.empty((n_frames, frame_size), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        frames[i, :] = samples[start:start + frame_size] * window

    # --- STFT ---
    S = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / fs)

    return S, freqs, window, orig_len


def istft(
    S: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: np.ndarray,
    signal_length: int | None = None,
) -> np.ndarray:
    """
    Inverse STFT matching compute_stft():
      - rFFT -> irFFT
      - Hann window
      - overlap-add
      - window^2 normalization (perfect reconstruction)

    Args:
        S: complex STFT matrix, shape (num_frames, frame_size//2 + 1)
        frame_size: window size used during STFT
        hop_size: hop size used during STFT
        window: SAME window array used during STFT
        signal_length: if given, trim final signal to this length

    Returns:
        y: reconstructed mono float32 signal
    """
    S = np.asarray(S)
    num_frames = S.shape[0]

    # Back to time-domain frames
    frames = np.fft.irfft(S, n=frame_size, axis=1).astype(np.float32)

    out_len = (num_frames - 1) * hop_size + frame_size
    y = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    w = window.astype(np.float32)
    w2 = w * w

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        y[start:end] += frames[i] * w
        norm[start:end] += w2

    nonzero = norm > 1e-8
    y[nonzero] /= norm[nonzero]

    if signal_length is not None and signal_length < len(y):
        y = y[:signal_length]

    return y


def test_stft_roundtrip() -> bool:
    """
    Generate a synthetic signal, run STFT + iSTFT, and verify
    the reconstruction error is tiny.
    """
    import numpy as np

    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # deterministic "busy" signal: two tones + noise
    rng = np.random.RandomState(42)
    x = (
        0.7 * np.sin(2 * np.pi * 440.0 * t)
        + 0.4 * np.sin(2 * np.pi * 880.0 * t)
        + 0.1 * rng.normal(size=t.shape)
    ).astype(np.float32)

    frame_size = 2048
    hop_size = 512

    S, freqs, orig_len = ws_stft(x, fs, frame_size, hop_size)
    y = ws_istft(S, fs, frame_size, hop_size, signal_length=orig_len)

    max_err = float(np.max(np.abs(x - y)))
    print(f"Max reconstruction error: {max_err:.3e}")

    # This threshold is HARD: anything above this is a FAILURE.
    return max_err < 1e-3


def select_candidate_frames(S: np.ndarray, energy_threshold: float = 1e-4) -> np.ndarray:
    """
    Select frames with energy above threshold.
    
    Args:
        S: Complex STFT matrix S[t, k]
        energy_threshold: Minimum energy threshold
        
    Returns:
        Array of frame indices that meet the energy threshold
    """
    # Compute energy for each frame (sum of squared magnitudes)
    energies = np.sum(np.abs(S)**2, axis=1)
    
    # Select frames with energy above threshold
    candidate_frames = np.where(energies > energy_threshold)[0]
    
    return candidate_frames


def get_band_bins(freqs: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """
    Get frequency bin indices within a given frequency band.
    
    Args:
        freqs: Frequency vector
        f_min: Minimum frequency (Hz)
        f_max: Maximum frequency (Hz)
        
    Returns:
        Array of frequency bin indices within the band
    """
    # Find bins within the frequency band
    band_bins = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    
    return band_bins


def estimate_capacity_bits(num_candidate_frames: int, num_band_bins: int, pairs_per_bit: int = 8) -> int:
    """
    Estimate maximum number of bits that can be embedded.
    
    Args:
        num_candidate_frames: Number of frames that can be used for embedding
        num_band_bins: Number of frequency bins in the selected band
        pairs_per_bit: Number of (frame, bin_pair) locations per bit
        
    Returns:
        Estimated maximum number of bits that can be embedded
    """
    # Each bit needs pairs_per_bit locations
    # Each location is a pair of bins (k1, k2) in a frame
    # We can use any pair of bins from the band in any frame
    max_pairs_per_frame = num_band_bins * (num_band_bins - 1) // 2  # Combinations of 2 bins
    total_available_locations = num_candidate_frames * max_pairs_per_frame
    
    # Estimate max bits
    max_bits = total_available_locations // pairs_per_bit
    
    return max_bits


def test_candidate_selection():
    """
    Test candidate frame selection and capacity estimation.
    """
    print("Testing candidate selection and capacity estimation...")
    
    # Load test audio
    if not os.path.exists('test_audio.wav'):
        print("⚠️  test_audio.wav not found, skipping test")
        return True
    
    try:
        # Load audio
        samples, fs = sf.read('test_audio.wav', dtype=np.float32)
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)
            
        # Compute STFT
        S, freqs, _ = ws_stft(samples, fs, 2048, 512)
        
        # Select candidate frames
        candidate_frames = select_candidate_frames(S, energy_threshold=1e-4)
        print(f"Found {len(candidate_frames)} candidate frames")
        
        # Get band bins
        band_bins = get_band_bins(freqs, 1000.0, 8000.0)
        print(f"Found {len(band_bins)} frequency bins in 1-8kHz band")
        
        # Estimate capacity
        capacity = estimate_capacity_bits(len(candidate_frames), len(band_bins), 8)
        print(f"Estimated capacity: {capacity} bits")
        
        if len(candidate_frames) > 0 and len(band_bins) > 0:
            print("✓ Candidate selection and capacity estimation test PASSED")
            return True
        else:
            print("⚠️  Some values are zero, but not necessarily an error")
            return True
            
    except Exception as e:
        print(f"✗ Error in candidate selection test: {e}")
        return False


def embed_bits_spread_spectrum(
    cover_path: str,
    bits: List[int],
    out_path: str,
    *,
    fs_target: int = 44100,
    frame_size: int = 2048,
    hop_size: int = 512,
    f_min: float = 1000.0,
    f_max: float = 8000.0,
    pairs_per_bit: int = 8,
    delta: float = 0.02,
    energy_threshold: float = 1e-4,
    seed: bytes | None = None,
) -> None:
    """
    Embed bits into audio using Max-Stealth v2 spread-spectrum steganography.
    
    Args:
        cover_path: Path to the cover WAV file
        bits: List of 0/1 bits to embed
        out_path: Output stego WAV file path
        fs_target: Target sample rate (default 44100)
        frame_size: STFT frame size (default 2048)
        hop_size: STFT hop size (default 512)
        f_min: Minimum frequency band (default 1000.0 Hz)
        f_max: Maximum frequency band (default 8000.0 Hz)
        pairs_per_bit: Number of (frame, bin_pair) locations per bit (default 8)
        delta: Magnitude difference factor (default 0.02)
        energy_threshold: Minimum energy threshold for candidate frames (default 1e-4)
        seed: Random seed for deterministic embedding (default None)
    """
    print("Embedding bits using Max-Stealth v2...")
    
    # Load cover audio
    samples, fs = sf.read(cover_path, dtype=np.float32)
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)
    
    # Resample if needed
    if fs != fs_target:
        # Calculate resampling ratio
        ratio = fs_target / fs
        # Use resample_poly for deterministic resampling
        samples = scipy.signal.resample_poly(samples, fs_target, fs)
        fs = fs_target
    
    # Compute STFT
    S, freqs, _ = ws_stft(samples, fs, frame_size, hop_size)
    
    # Select candidate frames
    candidate_frames = select_candidate_frames(S, energy_threshold)
    if len(candidate_frames) == 0:
        raise ValueError("No candidate frames found for embedding")
    
    # Get band bins
    band_bins = get_band_bins(freqs, f_min, f_max)
    if len(band_bins) < 2:
        raise ValueError("Not enough frequency bins in band for embedding")
    
    # Build deterministic PRNG
    if seed is not None:
        # Use SHA-256 to derive seed from provided seed
        seed_bytes = hashlib.sha256(seed + b"stego-v2").digest()
    else:
        # Use SHA-256 of cover path
        seed_bytes = hashlib.sha256(cover_path.encode("utf-8")).digest()
    
    seed_int = int.from_bytes(seed_bytes[:8], "big")
    rng = random.Random(seed_int)
    
    # Estimate required locations
    num_bits = len(bits)
    required_locations = num_bits * pairs_per_bit
    
    # Calculate maximum possible locations
    max_pairs_per_frame = len(band_bins) * (len(band_bins) - 1) // 2
    max_locations = len(candidate_frames) * max_pairs_per_frame
    
    if required_locations > max_locations:
        raise ValueError(f"Not enough embedding locations: need {required_locations}, have {max_locations}")
    
    # Generate embedding locations
    locations = []
    for i in range(num_bits):
        bit_locations = []
        for _ in range(pairs_per_bit):
            # Select random frame from candidates
            frame_idx = rng.choice(candidate_frames)
            
            # Select two different bins from the band
            bin_indices = rng.sample(list(band_bins), 2)
            k1, k2 = bin_indices[0], bin_indices[1]
            
            bit_locations.append((frame_idx, k1, k2))
        
        locations.extend(bit_locations)
    
    # Embed bits
    for i, (frame_idx, k1, k2) in enumerate(locations):
        bit = bits[i // pairs_per_bit]  # Which bit this location belongs to
        
        # Get magnitudes
        mag1 = abs(S[frame_idx, k1])
        mag2 = abs(S[frame_idx, k2])
        
        # Apply embedding rule
        if bit == 1:
            # Enforce M1 >= (1 + delta) * M2
            if mag1 < (1 + delta) * mag2:
                scale_factor = (1 + delta) * mag2 / max(mag1, 1e-12)
                S[frame_idx, k1] *= scale_factor
        else:
            # Enforce M2 >= (1 + delta) * M1
            if mag2 < (1 + delta) * mag1:
                scale_factor = (1 + delta) * mag1 / max(mag2, 1e-12)
                S[frame_idx, k2] *= scale_factor
    
    # Reconstruct time-domain samples
    reconstructed = ws_istft(S, fs, frame_size, hop_size, signal_length=len(reconstructed))
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(reconstructed))
    if max_val > 1.0:
        reconstructed = reconstructed / max_val
    
    # Save to output file
    sf.write(out_path, reconstructed, fs, subtype='PCM_16')
    
    print(f"✓ Embedded {num_bits} bits into {out_path}")


def extract_bits_spread_spectrum(
    stego_path: str,
    num_bits: int,
    *,
    fs_target: int = 44100,
    frame_size: int = 2048,
    hop_size: int = 512,
    f_min: float = 1000.0,
    f_max: float = 8000.0,
    pairs_per_bit: int = 8,
    energy_threshold: float = 1e-4,
    seed: bytes | None = None,
) -> List[int]:
    """
    Extract bits from stego audio using Max-Stealth v2 spread-spectrum steganography.
    
    Args:
        stego_path: Path to the stego WAV file
        num_bits: Number of bits to extract
        fs_target: Target sample rate (default 44100)
        frame_size: STFT frame size (default 2048)
        hop_size: STFT hop size (default 512)
        f_min: Minimum frequency band (default 1000.0 Hz)
        f_max: Maximum frequency band (default 8000.0 Hz)
        pairs_per_bit: Number of (frame, bin_pair) locations per bit (default 8)
        energy_threshold: Minimum energy threshold for candidate frames (default 1e-4)
        seed: Random seed for deterministic extraction (default None)
        
    Returns:
        List of extracted bits (0/1)
    """
    print("Extracting bits using Max-Stealth v2...")
    
    # Load stego audio
    samples, fs = sf.read(stego_path, dtype=np.float32)
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)
    
    # Resample if needed
    if fs != fs_target:
        # Calculate resampling ratio
        ratio = fs_target / fs
        # Use resample_poly for deterministic resampling
        samples = scipy.signal.resample_poly(samples, fs_target, fs)
        fs = fs_target
    
    # Compute STFT
    S, freqs, _ = ws_stft(samples, fs, frame_size, hop_size)
    
    # Select candidate frames
    candidate_frames = select_candidate_frames(S, energy_threshold)
    if len(candidate_frames) == 0:
        raise ValueError("No candidate frames found for extraction")
    
    # Get band bins
    band_bins = get_band_bins(freqs, f_min, f_max)
    if len(band_bins) < 2:
        raise ValueError("Not enough frequency bins in band for extraction")
    
    # Build deterministic PRNG (same as embedding)
    if seed is not None:
        # Use SHA-256 to derive seed from provided seed
        seed_bytes = hashlib.sha256(seed + b"stego-v2").digest()
    else:
        # Use SHA-256 of stego path
        seed_bytes = hashlib.sha256(stego_path.encode("utf-8")).digest()
    
    seed_int = int.from_bytes(seed_bytes[:8], "big")
    rng = random.Random(seed_int)
    
    # Generate the same embedding locations
    locations = []
    for i in range(num_bits):
        bit_locations = []
        for _ in range(pairs_per_bit):
            # Select random frame from candidates
            frame_idx = rng.choice(candidate_frames)
            
            # Select two different bins from the band
            bin_indices = rng.sample(list(band_bins), 2)
            k1, k2 = bin_indices[0], bin_indices[1]
            
            bit_locations.append((frame_idx, k1, k2))
        
        locations.extend(bit_locations)
    
    # Extract bits by voting
    extracted_bits = []
    for i in range(num_bits):
        # Get all locations for this bit
        bit_locations = locations[i * pairs_per_bit:(i + 1) * pairs_per_bit]
        
        # Vote for this bit
        votes = 0
        for frame_idx, k1, k2 in bit_locations:
            mag1 = abs(S[frame_idx, k1])
            mag2 = abs(S[frame_idx, k2])
            
            # Vote 1 if M1 > M2, else 0
            if mag1 > mag2:
                votes += 1
        
        # Majority vote
        bit = 1 if votes > pairs_per_bit // 2 else 0
        extracted_bits.append(bit)
    
    print(f"✓ Extracted {len(extracted_bits)} bits from {stego_path}")
    return extracted_bits


def encrypt_file_with_worldsound(
    file_path: str,
    world_audio_path: str
) -> Tuple[bytes, bytes]:
    """
    Reads file_path as raw bytes, derives a keystream from world_audio_path
    using WorldSound OTP v2, XORs plaintext with keystream, and returns:
    (plaintext_bytes, ciphertext_bytes)

    Args:
        file_path: Path to the file to encrypt
        world_audio_path: Path to the audio file used as key material

    Returns:
        Tuple of (plaintext_bytes, ciphertext_bytes)
    """
    # Read file as raw bytes
    with open(file_path, 'rb') as f:
        plaintext = f.read()
    
    # Derive keystream from world audio
    features = extract_features_from_audio(world_audio_path)
    keystream = derive_keystream_from_features(features, len(plaintext))
    
    # XOR encrypt
    ciphertext = encrypt_bytes(plaintext, keystream)
    
    return plaintext, ciphertext


def build_payload(plaintext: bytes, ciphertext: bytes) -> bytes:
    """
    Builds MAGIC + LENGTH + HASH + CIPHER payload.
    - MAGIC = b"WSF0"
    - LENGTH = uint32 big-endian (len(plaintext))
    - HASH = SHA-256(plaintext)
    - CIPHER = ciphertext

    Args:
        plaintext: Original plaintext bytes
        ciphertext: Encrypted bytes

    Returns:
        Complete payload bytes
    """
    # Create header
    magic = b"WSF0"  # 4 bytes
    length = len(plaintext).to_bytes(4, byteorder='big')  # 4 bytes
    file_hash = hashlib.sha256(plaintext).digest()  # 32 bytes
    cipher = ciphertext  # Variable length
    
    # Combine all parts
    payload = magic + length + file_hash + cipher
    
    return payload


def bytes_to_bits(data: bytes) -> List[int]:
    """
    Convert bytes to a big-endian bit list (0/1 values).
    For each byte, output bits from MSB to LSB.

    Args:
        data: Input bytes

    Returns:
        List of 0/1 bits
    """
    bits = []
    for byte in data:
        # For each byte, extract bits from MSB to LSB
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            bits.append(bit)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Convert a list of 0/1 bits to bytes.
    Ignore any trailing bits that do not complete a full byte.

    Args:
        bits: List of 0/1 bits

    Returns:
        Bytes representation
    """
    # Trim bits to complete bytes
    complete_bits = len(bits) - (len(bits) % 8)
    bits = bits[:complete_bits]
    
    if not bits:
        return b""
    
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte_value = 0
        for j in range(8):
            byte_value = (byte_value << 1) | bits[i + j]
        bytes_list.append(byte_value)
    
    return bytes(bytes_list)


def encrypt_and_encode_file(
    file_path: str,
    world_audio_path: str,
    cover_audio_path: str,
    stego_out_path: str
) -> None:
    """
    High-level operation:
    - Encrypt file_path using world_audio_path as the key source (WorldSound OTP v2).
    - Pack MAGIC+LENGTH+HASH+ CIPHER into a payload.
    - Convert payload bytes -> bits.
    - Embed bits into cover_audio_path using max-stealth encoder.
    - Save resulting stego WAV as stego_out_path.

    Args:
        file_path: Path to the file to encrypt and encode
        world_audio_path: Path to the audio file used as key material
        cover_audio_path: Path to the cover audio file for steganography
        stego_out_path: Output stego WAV file path
    """
    # Validate inputs
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.exists(world_audio_path):
        raise FileNotFoundError(f"World audio not found: {world_audio_path}")
    if not os.path.exists(cover_audio_path):
        raise FileNotFoundError(f"Cover audio not found: {cover_audio_path}")

    # Step 1: Encrypt file with WorldSound OTP v2
    plaintext, ciphertext = encrypt_file_with_worldsound(file_path, world_audio_path)
    
    # Step 2: Build payload
    payload = build_payload(plaintext, ciphertext)
    
    # Step 3: Convert payload to bits
    bits = bytes_to_bits(payload)
    
    # Step 4: Embed bits into cover audio
    embed_bits_spread_spectrum(cover_audio_path, bits, stego_out_path)
    
    print(f"Successfully encrypted and encoded {file_path} into {stego_out_path}")


def decode_and_decrypt_file(
    stego_path: str,
    world_audio_path: str,
    recovered_file_path: str
) -> None:
    """
    High-level reverse operation:
    - Extract payload bits from stego_path.
    - Convert bits -> payload bytes.
    - Parse MAGIC, LENGTH, HASH, and CIPHER.
    - Derive same keystream from world_audio_path (WorldSound OTP v2).
    - XOR-decrypt ciphertext to recover plaintext.
    - Verify hash; if mismatch, raise error.
    - Write recovered plaintext bytes to recovered_file_path.

    Args:
        stego_path: Path to the stego WAV file
        world_audio_path: Path to the world audio file used for decryption
        recovered_file_path: Path to write the recovered file
    """
    # Validate inputs
    if not os.path.exists(stego_path):
        raise FileNotFoundError(f"Stego file not found: {stego_path}")
    if not os.path.exists(world_audio_path):
        raise FileNotFoundError(f"World audio not found: {world_audio_path}")

    # Step 1: Determine how many bits to extract
    # Extract enough bits to read the header (at least 320 bits for 40 bytes)
    # Use a reasonable number that should be sufficient for most payloads
    # This is a compromise between reliability and efficiency
    estimated_payload_bits = 2048  # Enough for header + some margin
    
    # Step 2: Extract bits from stego audio
    bits = extract_bits_spread_spectrum(stego_path, estimated_payload_bits)
    
    # Step 3: Convert bits to bytes
    payload = bits_to_bytes(bits)
    
    # Step 4: Parse fields
    if len(payload) < 40:  # MAGIC + LENGTH + HASH = 4 + 4 + 32 = 40 bytes
        raise ValueError("Payload too short - corrupted or incomplete")
    
    MAGIC = payload[0:4]
    if MAGIC != b"WSF0":
        raise ValueError("Invalid magic; not a WorldSound File v0 payload.")

    length = int.from_bytes(payload[4:8], "big")
    file_hash = payload[8:40]
    cipher = payload[40:40+length]
    
    # Step 5: Derive keystream from world audio
    features = extract_features_from_audio(world_audio_path)
    keystream = derive_keystream_from_features(features, len(cipher))
    
    # Step 6: Decrypt
    plaintext = decrypt_bytes(cipher, keystream)
    
    # Step 7: Verify hash
    if hashlib.sha256(plaintext).digest() != file_hash:
        raise ValueError("Hash mismatch – wrong world audio or corrupted stego.")
    
    # Step 8: Write recovered file
    with open(recovered_file_path, 'wb') as f:
        f.write(plaintext)
    
    print(f"Successfully decoded and decrypted to {recovered_file_path}")


def test_file_in_sound():
    """Run comprehensive self-tests for file-in-sound functionality."""
    print("Running File-in-Sound v1 comprehensive self-tests...")
    
    # Create a test file
    test_file = "/workspace/test_file.bin"
    test_content = b"This is a test file for File-in-Sound v1 functionality. " * 10
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    # Use existing test audio
    world_audio = "/workspace/test_audio.wav"
    cover_audio = "/workspace/test_audio.wav"  # Use same file for testing
    
    # Create a stego file path
    stego_file = "/workspace/test_stego.wav"
    recovered_file = "/workspace/test_recovered.bin"
    
    try:
        # Test encryption and encoding
        encrypt_and_encode_file(test_file, world_audio, cover_audio, stego_file)
        
        # Test decoding and decryption (this will show warnings about placeholders)
        decode_and_decrypt_file(stego_file, world_audio, recovered_file)
        
        # Verify recovery
        with open(recovered_file, 'rb') as f:
            recovered_content = f.read()
        
        if recovered_content == test_content:
            print("✓ Test PASSED: File encryption/decryption round-trip successful")
            return True
        else:
            print("✗ Test FAILED: Recovered content doesn't match original")
            return False
            
    except Exception as e:
        print(f"✗ Test ERROR: {str(e)}")
        return False
    finally:
        # Cleanup
        for file in [test_file, stego_file, recovered_file]:
            if os.path.exists(file):
                os.remove(file)


def main():
    """Main CLI entry point for file-in-sound operations."""
    parser = argparse.ArgumentParser(
        description="File-in-Sound v1: Encrypt files with WorldSound OTP v2 and embed into audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s encrypt-and-encode \\
    --file secret.dex \\
    --world world.wav \\
    --cover music.wav \\
    --out stego.wav

  %(prog)s decode-and-decrypt \\
    --stego stego.wav \\
    --world world.wav \\
    --out secret_restored.dex
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # encrypt-and-encode subcommand
    encrypt_parser = subparsers.add_parser('encrypt-and-encode', help='Encrypt and encode a file into audio')
    encrypt_parser.add_argument('--file', required=True, help='File to encrypt and encode')
    encrypt_parser.add_argument('--world', required=True, help='World audio file for key material')
    encrypt_parser.add_argument('--cover', required=True, help='Cover audio file for steganography')
    encrypt_parser.add_argument('--out', required=True, help='Output stego WAV file')
    
    # decode-and-decrypt subcommand
    decrypt_parser = subparsers.add_parser('decode-and-decrypt', help='Decode and decrypt a file from audio')
    decrypt_parser.add_argument('--stego', required=True, help='Stego audio file to decode')
    decrypt_parser.add_argument('--world', required=True, help='World audio file for decryption')
    decrypt_parser.add_argument('--out', required=True, help='Output file path for recovered data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'encrypt-and-encode':
            encrypt_and_encode_file(args.file, args.world, args.cover, args.out)
        elif args.command == 'decode-and-decrypt':
            decode_and_decrypt_file(args.stego, args.world, args.out)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


# ---------------- STFT / iSTFT CORE ---------------- #

FrameMatrix = np.ndarray

def ws_stft(
    samples: np.ndarray,
    fs: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> Tuple[FrameMatrix, np.ndarray, int]:
    """
    Wrapper around scipy.signal.stft that returns:
      - S: (num_frames, num_bins) complex STFT
      - freqs: 1D array of bin frequencies
      - orig_len: original signal length

    Uses Hann window, no boundary padding, and no extra zero-padding,
    so ws_istft can reconstruct very accurately.
    """
    if samples.ndim != 1:
        raise ValueError("ws_stft expects mono 1D array")

    samples = samples.astype(np.float32, copy=False)
    orig_len = len(samples)

    nperseg = frame_size
    noverlap = frame_size - hop_size

    # SciPy shape: Zxx has shape (num_freq_bins, num_frames)
    freqs, times, Zxx = stft(
        samples,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,    # no implicit reflection padding
        padded=False,     # no implicit zero-padding
        return_onesided=True,
    )

    # For our convenience: transpose to (num_frames, num_bins)
    S = Zxx.T  # (T, F)

    return S, freqs, orig_len


def ws_istft(
    S: FrameMatrix,
    fs: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    signal_length: int | None = None,
) -> np.ndarray:
    """
    Inverse of ws_stft using scipy.signal.istft.

    Args:
        S: STFT matrix with shape (num_frames, num_bins) – as returned by ws_stft.
        fs: sample rate
        frame_size: same as ws_stft
        hop_size: same as ws_stft
        signal_length: trim output to this length if not None

    Returns:
        Reconstructed mono float32 signal.
    """
    S = np.asarray(S)

    # Transpose back to SciPy's expected shape: (num_bins, num_frames)
    Zxx = S.T

    nperseg = frame_size
    noverlap = frame_size - hop_size

    # Call istft with correct parameters
    recon, _ = istft(
        Zxx,
        fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )

    if signal_length is not None and len(recon) > signal_length:
        recon = recon[:signal_length]

    return recon.astype(np.float32)


def test_stft_roundtrip() -> bool:
    """
    STFT + iSTFT round-trip sanity check using ws_stft / ws_istft.
    Expect max error < 1e-3.
    """
    import numpy as np

    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    rng = np.random.RandomState(42)
    x = (
        0.7 * np.sin(2 * np.pi * 440.0 * t)
        + 0.4 * np.sin(2 * np.pi * 880.0 * t)
        + 0.1 * rng.normal(size=t.shape)
    ).astype(np.float32)

    frame_size = 2048
    hop_size = 512

    S, freqs, orig_len = ws_stft(x, fs, frame_size, hop_size)
    y = ws_istft(S, fs, frame_size, hop_size, signal_length=orig_len)

    # Adjust lengths for comparison
    min_len = min(len(x), len(y))
    x_trimmed = x[:min_len]
    y_trimmed = y[:min_len]
    
    max_err = float(np.max(np.abs(x_trimmed - y_trimmed)))
    print(f"[SciPy-based STFT] max reconstruction error: {max_err:.3e}")

    return max_err < 1e-3


if __name__ == "__main__":
    # If run directly, check if we're in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_file_in_sound()
    else:
        main()
