
#!/usr/bin/env python3
"""
WorldSound OTP - Version 2
Deterministic audio-based keystream generator and XOR-based encrypt/decrypt tool.

This tool converts audio characteristics into a cryptographic keystream using
multiple spectral features extracted from an input audio file.
"""

import argparse
import hashlib
import numpy as np
import scipy.signal
import soundfile as sf
from typing import Tuple, Optional, Union
import sys


def load_audio_mono(audio_path: str, target_fs: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert to mono if needed.
    
    Args:
        audio_path: Path to the audio file
        target_fs: Target sample rate (default 44100 Hz)
        
    Returns:
        Tuple of (samples, fs) where samples is a float32 numpy array in range [-1.0, 1.0]
        and fs is the sample rate
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: For other audio loading errors
    """
    try:
        # Load audio file
        samples, fs = sf.read(audio_path, dtype=np.float32)
        
        # Convert to mono if stereo or multi-channel
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)
            
        # Resample if needed
        if fs != target_fs:
            # Calculate resampling ratio
            ratio = target_fs / fs
            # Use resample_poly for deterministic resampling
            samples = scipy.signal.resample_poly(samples, target_fs, fs)
            fs = target_fs
            
        return samples, fs
    except Exception as e:
        raise Exception(f"Error loading audio file {audio_path}: {str(e)}")


def compute_stft_frames(samples: np.ndarray, fs: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT frames from audio samples.
    
    Args:
        samples: Audio samples (float32 numpy array in range [-1.0, 1.0])
        fs: Sample rate (default 44100 Hz)
        
    Returns:
        Tuple of (frames, freqs) where:
        - frames: shape (num_frames, 2048) windowed time-domain frames
        - freqs: frequency vector from rfftfreq
    """
    # STFT parameters
    N = 2048  # Window size
    H = 512   # Hop size (75% overlap)
    
    # Create Hann window
    window = scipy.signal.windows.hann(N, sym=False)
    
    # Compute STFT frames
    num_frames = 1 + (len(samples) - N) // H
    frames = np.zeros((num_frames, N))
    
    for i in range(num_frames):
        start = i * H
        end = start + N
        if end <= len(samples):
            frames[i] = samples[start:end] * window
        else:
            # Handle last frame if it's shorter
            frame_data = samples[start:]
            padded_frame = np.zeros(N)
            padded_frame[:len(frame_data)] = frame_data
            frames[i] = padded_frame * window
    
    # Compute frequency vector
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    
    return frames, freqs


def extract_features_from_audio(audio_path: str) -> bytes:
    """
    Extract multiple spectral features from audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Bytes representing extracted features from the audio
        
    Raises:
        Exception: For audio processing errors
    """
    # Load audio
    samples, fs = load_audio_mono(audio_path)
    
    # Compute STFT frames
    frames, freqs = compute_stft_frames(samples, fs)
    
    # Feature extraction parameters
    f_min = 100.0
    f_max = 12000.0
    
    # Find frequency indices for the band
    freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    if len(freq_indices) == 0:
        # If no frequencies in band, use all frequencies
        freq_indices = np.arange(len(freqs))
    
    num_band_bins = len(freq_indices)
    
    # Collect features for all frames
    feature_bytes = bytearray()
    
    # Previous frame data for flux calculation
    prev_mag_band = None
    
    for i, frame in enumerate(frames):
        # Compute FFT
        X = np.fft.rfft(frame)
        mag = np.abs(X)
        phase = np.angle(X)
        
        # Extract band data
        if num_band_bins > 0:
            mag_band = mag[freq_indices]
            phase_band = phase[freq_indices]
            freqs_band = freqs[freq_indices]
        else:
            mag_band = np.array([])
            phase_band = np.array([])
            freqs_band = np.array([])
        
        # Feature a) Dominant frequency index
        if len(mag_band) > 0 and np.sum(mag_band) > 0:
            k_peak = np.argmax(mag_band)
            bin_ratio = k_peak / max(1, num_band_bins - 1)
            dom_bin_q = int(bin_ratio * 255)
        else:
            dom_bin_q = 0
        
        # Feature b) Spectral centroid
        if len(mag_band) > 0 and np.sum(mag_band) > 0:
            centroid = np.sum(freqs_band * mag_band) / np.sum(mag_band)
            centroid_ratio = (centroid - f_min) / max(1, f_max - f_min)
            centroid_ratio = np.clip(centroid_ratio, 0.0, 1.0)
            centroid_q = int(centroid_ratio * 255)
        else:
            centroid_q = 0
        
        # Feature c) Log-energy
        if len(frame) > 0:
            E = np.sum(frame**2) / len(frame)
            logE = np.log10(E + 1e-12)
            logE_clamped = np.clip((logE + 12.0) / 12.0, 0.0, 1.0)
            logE_q = int(logE_clamped * 255)
        else:
            logE_q = 0
        
        # Feature d) Spectral flux
        if prev_mag_band is not None and len(mag_band) > 0 and len(prev_mag_band) > 0:
            flux = np.sum((mag_band - prev_mag_band)**2) / (np.sum(prev_mag_band**2) + 1e-12)
            flux_norm = flux / (flux + 1.0)
            flux_norm = np.clip(flux_norm, 0.0, 1.0)
            flux_q = int(flux_norm * 255)
        else:
            flux_q = 0
        
        # Feature e) Phase difference
        if prev_mag_band is not None and len(phase_band) > 0:
            phase_diff = phase_band - prev_phase_band
            # Wrap phase differences to [-pi, pi]
            phase_diff = np.angle(np.exp(1j * phase_diff))
            mean_abs_phase_diff = np.mean(np.abs(phase_diff))
            phase_ratio = mean_abs_phase_diff / np.pi
            phase_ratio = np.clip(phase_ratio, 0.0, 1.0)
            phase_q = int(phase_ratio * 255)
        else:
            phase_q = 0
        
        # Store current data for next iteration
        prev_mag_band = mag_band
        prev_phase_band = phase_band
        
        # Pack features into 5 bytes
        feature_bytes.extend([
            dom_bin_q,
            centroid_q,
            logE_q,
            flux_q,
            phase_q
        ])
    
    return bytes(feature_bytes)


def derive_seed_from_features(features: bytes) -> bytes:
    """
    Derive a 32-byte seed from feature bytes using SHA-256.
    
    Args:
        features: Feature bytes extracted from audio
        
    Returns:
        32-byte seed derived from features using SHA-256
    """
    return hashlib.sha256(features).digest()


def derive_keystream_from_features(features: bytes, length: int) -> bytes:
    """
    Generate a deterministic keystream from audio features.
    
    Args:
        features: Feature bytes extracted from audio
        length: Number of bytes to generate
        
    Returns:
        Deterministic keystream of specified length
    """
    seed = derive_seed_from_features(features)
    
    # Use hash-based stream cipher (counter mode with SHA-256)
    out = bytearray()
    counter = 0
    
    while len(out) < length:
        counter_bytes = counter.to_bytes(4, byteorder="big", signed=False)
        block = hashlib.sha256(seed + counter_bytes).digest()
        out.extend(block)
        counter += 1
    
    return bytes(out[:length])


def encrypt_bytes(plaintext: bytes, keystream: bytes) -> bytes:
    """
    Encrypt plaintext using XOR with keystream.
    
    Args:
        plaintext: Plaintext bytes to encrypt
        keystream: Keystream bytes (must be at least as long as plaintext)
        
    Returns:
        Ciphertext bytes
    """
    return bytes(p ^ k for p, k in zip(plaintext, keystream))


def decrypt_bytes(ciphertext: bytes, keystream: bytes) -> bytes:
    """
    Decrypt ciphertext using XOR with keystream.
    
    Args:
        ciphertext: Ciphertext bytes to decrypt
        keystream: Keystream bytes (must be at least as long as ciphertext)
        
    Returns:
        Plaintext bytes
    """
    return bytes(c ^ k for c, k in zip(ciphertext, keystream))


def encrypt_with_world_sound(audio_path: str, plaintext: bytes) -> bytes:
    """
    Encrypt plaintext using audio features as keystream source.
    
    Args:
        audio_path: Path to the audio file
        plaintext: Plaintext bytes to encrypt
        
    Returns:
        Ciphertext bytes
    """
    features = extract_features_from_audio(audio_path)
    keystream = derive_keystream_from_features(features, len(plaintext))
    return encrypt_bytes(plaintext, keystream)


def decrypt_with_world_sound(audio_path: str, ciphertext: bytes) -> bytes:
    """
    Decrypt ciphertext using audio features as keystream source.
    
    Args:
        audio_path: Path to the audio file
        ciphertext: Ciphertext bytes to decrypt
        
    Returns:
        Plaintext bytes
    """
    features = extract_features_from_audio(audio_path)
    keystream = derive_keystream_from_features(features, len(ciphertext))
    return decrypt_bytes(ciphertext, keystream)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WorldSound OTP - Version 2: Audio-based deterministic encryption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s extract-key world.wav --hex
  %(prog)s encrypt world.wav input.txt output.bin
  %(prog)s decrypt world.wav input.bin output.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # extract-key subcommand
    extract_parser = subparsers.add_parser('extract-key', help='Extract feature bytes or seed from audio')
    extract_parser.add_argument('audio_path', help='Path to the audio file')
    extract_parser.add_argument('--mode', choices=['features', 'seed'], default='features',
                               help='What to extract (default: features)')
    extract_parser.add_argument('--out', help='Output file path')
    extract_parser.add_argument('--hex', action='store_true', help='Print output in hex format')
    
    # encrypt subcommand
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a file using audio features')
    encrypt_parser.add_argument('audio_path', help='Path to the audio file')
    encrypt_parser.add_argument('input_file', help='Input text file to encrypt')
    encrypt_parser.add_argument('output_file', help='Output binary file for encrypted data')
    encrypt_parser.add_argument('--raw', action='store_true', help='Treat input as raw bytes instead of UTF-8 text')
    
    # decrypt subcommand
    decrypt_parser = subparsers.add_parser('decrypt', help='Decrypt a file using audio features')
    decrypt_parser.add_argument('audio_path', help='Path to the audio file')
    decrypt_parser.add_argument('input_file', help='Input binary file to decrypt')
    decrypt_parser.add_argument('output_file', help='Output text file for decrypted data')
    decrypt_parser.add_argument('--raw', action='store_true', help='Write output as raw bytes instead of UTF-8 text')
    
    # Add file-in-sound subcommands if the module is available
    try:
        from worldsound_file_tools import (
            encrypt_and_encode_file, 
            decode_and_decrypt_file
        )
        
        # encrypt-and-encode subcommand
        encrypt_encode_parser = subparsers.add_parser('encrypt-and-encode', help='Encrypt and encode a file into audio')
        encrypt_encode_parser.add_argument('--file', required=True, help='File to encrypt and encode')
        encrypt_encode_parser.add_argument('--world', required=True, help='World audio file for key material')
        encrypt_encode_parser.add_argument('--cover', required=True, help='Cover audio file for steganography')
        encrypt_encode_parser.add_argument('--out', required=True, help='Output stego WAV file')
        
        # decode-and-decrypt subcommand
        decode_decrypt_parser = subparsers.add_parser('decode-and-decrypt', help='Decode and decrypt a file from audio')
        decode_decrypt_parser.add_argument('--stego', required=True, help='Stego audio file to decode')
        decode_decrypt_parser.add_argument('--world', required=True, help='World audio file for decryption')
        decode_decrypt_parser.add_argument('--out', required=True, help='Output file path for recovered data')
        
    except ImportError:
        # If file tools aren't available, skip adding these subcommands
        pass
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'extract-key':
            features = extract_features_from_audio(args.audio_path)
            
            if args.mode == 'seed':
                output = derive_seed_from_features(features)
            else:
                output = features
                
            if args.out:
                with open(args.out, 'wb') as f:
                    f.write(output)
                print(f"Saved {len(output)} bytes to {args.out}")
            elif args.hex:
                print(output.hex())
            else:
                print(f"Extracted {len(output)} bytes")
                
        elif args.command == 'encrypt':
            try:
                with open(args.input_file, 'rb' if args.raw else 'r', encoding=None if args.raw else 'utf-8') as f:
                    if args.raw:
                        plaintext = f.read()
                    else:
                        plaintext = f.read().encode('utf-8')
            except FileNotFoundError:
                print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading input file: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
            ciphertext = encrypt_with_world_sound(args.audio_path, plaintext)
            
            try:
                with open(args.output_file, 'wb') as f:
                    f.write(ciphertext)
                print(f"Encrypted {len(plaintext)} bytes to {args.output_file}")
            except Exception as e:
                print(f"Error writing output file: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
        elif args.command == 'decrypt':
            try:
                with open(args.input_file, 'rb') as f:
                    ciphertext = f.read()
            except FileNotFoundError:
                print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading input file: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
            plaintext = decrypt_with_world_sound(args.audio_path, ciphertext)
            
            try:
                with open(args.output_file, 'wb' if args.raw else 'w', encoding=None if args.raw else 'utf-8') as f:
                    if args.raw:
                        f.write(plaintext)
                    else:
                        f.write(plaintext.decode('utf-8'))
                print(f"Decrypted {len(ciphertext)} bytes to {args.output_file}")
            except Exception as e:
                print(f"Error writing output file: {str(e)}", file=sys.stderr)
                sys.exit(1)
                
        elif args.command == 'encrypt-and-encode':
            # Import here to avoid circular dependencies
            from worldsound_file_tools import encrypt_and_encode_file
            encrypt_and_encode_file(args.file, args.world, args.cover, args.out)
            
        elif args.command == 'decode-and-decrypt':
            # Import here to avoid circular dependencies
            from worldsound_file_tools import decode_and_decrypt_file
            decode_and_decrypt_file(args.stego, args.world, args.out)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def test_worldsound():
    """Run comprehensive self-tests."""
    print("Running WorldSound OTP v2 comprehensive self-tests...")
    
    # Create a simple test audio file (silence) for testing
    test_audio_path = "/workspace/test_audio.wav"
    
    # Create a simple sine wave for testing
    fs = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Create a simple tone
    tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Save as WAV file
    sf.write(test_audio_path, tone, fs)
    
    # Test 1: Basic round-trip
    test_message = b"Hello from WorldSound OTP v2"
    try:
        # Encrypt
        ciphertext = encrypt_with_world_sound(test_audio_path, test_message)
        
        # Decrypt
        plaintext = decrypt_with_world_sound(test_audio_path, ciphertext)
        
        # Verify
        if plaintext == test_message:
            print("✓ Test 1 PASSED: Basic encryption/decryption round-trip successful")
        else:
            print("✗ Test 1 FAILED: Decrypted message doesn't match original")
            return False
    except Exception as e:
        print(f"✗ Test 1 ERROR: {str(e)}")
        return False
    
    # Test 2: Unicode/Emoji support
    unicode_message = "Hello 🌍! 你好世界! 🚀".encode('utf-8')
    try:
        # Encrypt
        ciphertext = encrypt_with_world_sound(test_audio_path, unicode_message)
        
        # Decrypt
        plaintext = decrypt_with_world_sound(test_audio_path, ciphertext)
        
        # Verify
        if plaintext == unicode_message:
            print("✓ Test 2 PASSED: Unicode/Emoji encryption/decryption successful")
        else:
            print("✗ Test 2 FAILED: Decrypted Unicode message doesn't match original")
            return False
    except Exception as e:
        print(f"✗ Test 2 ERROR: {str(e)}")
        return False
    
    # Test 3: Determinism - Generate keystream twice and compare
    try:
        features = extract_features_from_audio(test_audio_path)
        keystream1 = derive_keystream_from_features(features, 100)
        keystream2 = derive_keystream_from_features(features, 100)
        
        if keystream1 == keystream2:
            print("✓ Test 3 PASSED: Keystream generation is deterministic")
        else:
            print("✗ Test 3 FAILED: Keystream generation is not deterministic")
            return False
    except Exception as e:
        print(f"✗ Test 3 ERROR: {str(e)}")
        return False
    
    # Test 4: Empty message
    try:
        empty_ciphertext = encrypt_with_world_sound(test_audio_path, b"")
        empty_plaintext = decrypt_with_world_sound(test_audio_path, empty_ciphertext)
        
        if empty_plaintext == b"":
            print("✓ Test 4 PASSED: Empty message handling successful")
        else:
            print("✗ Test 4 FAILED: Empty message handling failed")
            return False
    except Exception as e:
        print(f"✗ Test 4 ERROR: {str(e)}")
        return False
    
    print("✓ All self-tests PASSED!")
    return True


if __name__ == "__main__":
    # Check if we're running the test
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_worldsound()
    else:
        main()
