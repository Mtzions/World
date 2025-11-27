
#!/usr/bin/env python3
"""
Debug-focused CLI script for WorldSound OTP v2 round-trip testing.
This script performs a complete encryption/decryption cycle with detailed logging.
"""

import argparse
import hashlib
import sys
from typing import Optional

# Import WorldSound OTP v2 functions
from worldsound import (
    extract_features_from_audio,
    derive_keystream_from_features,
    encrypt_bytes,
    decrypt_bytes,
    encrypt_with_world_sound,
    decrypt_with_world_sound
)


def format_hex_head(data: bytes, max_bytes: int = 32) -> str:
    """Format first N bytes of data as hex string."""
    if len(data) <= max_bytes:
        return ' '.join(f'{b:02x}' for b in data)
    else:
        head = data[:max_bytes]
        return ' '.join(f'{b:02x}' for b in head) + f' ... ({len(data)} bytes total)'


def safe_preview_text(text: str, max_chars: int = 200) -> str:
    """Create a safe preview of text that handles special characters."""
    if len(text) <= max_chars:
        return repr(text)
    else:
        return repr(text[:max_chars]) + f' ... ({len(text)} chars total)'


def run_roundtrip(
    world_audio_path: str,
    input_path: str,
    cipher_path: str,
    output_path: str,
    raw: bool
) -> None:
    """Perform a complete round-trip encryption/decryption test."""
    
    print("=" * 50)
    print("WorldSound OTP v2 – Debug Round Trip")
    print("=" * 50)
    print()
    
    print("Input Parameters:")
    print(f"    World audio:      {world_audio_path}")
    print(f"    Input text file:  {input_path}")
    print(f"    Cipher file:      {cipher_path}")
    print(f"    Output file:      {output_path}")
    print(f"    Mode:             {'RAW' if raw else 'TEXT'}")
    print()
    
    # 1) LOAD PLAINTEXT
    print("[1] Loading plaintext file")
    try:
        if not raw:
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                plaintext_text = f.read()
            plaintext_bytes = plaintext_text.encode("utf-8")
            print(f"    Path     : {input_path}")
            print(f"    Length   : {len(plaintext_bytes)} bytes")
            print(f"    Preview  : {safe_preview_text(plaintext_text, 120)}")
        else:
            with open(input_path, "rb") as f:
                plaintext_bytes = f.read()
            print(f"    Path     : {input_path}")
            print(f"    Length   : {len(plaintext_bytes)} bytes")
            print(f"    Hex head : {format_hex_head(plaintext_bytes, 32)}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}")
        sys.exit(1)
    
    # 2) EXTRACT FEATURES FROM WORLD AUDIO
    print("[2] Extracting WorldSound features")
    try:
        features = extract_features_from_audio(world_audio_path)
        features_hash = hashlib.sha256(features).digest()
        features_hash_hex = features_hash.hex()
        print(f"    Audio path      : {world_audio_path}")
        print(f"    Feature length  : {len(features)} bytes")
        print(f"    Feature SHA-256 : {features_hash_hex}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to extract features from audio: {e}")
        sys.exit(1)
    
    # 3) DERIVE KEYSTREAM
    print("[3] Deriving keystream")
    try:
        keystream = derive_keystream_from_features(features, len(plaintext_bytes))
        keystream_hash = hashlib.sha256(keystream).digest()
        keystream_hash_hex = keystream_hash.hex()
        print(f"    Keystream length: {len(keystream)} bytes")
        print(f"    Keystream SHA-256: {keystream_hash_hex}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to derive keystream: {e}")
        sys.exit(1)
    
    # 4) ENCRYPT
    print("[4] Encrypting plaintext -> ciphertext")
    try:
        ciphertext = encrypt_bytes(plaintext_bytes, keystream)
        with open(cipher_path, "wb") as f:
            f.write(ciphertext)
        
        cipher_hash = hashlib.sha256(ciphertext).digest()
        cipher_hash_hex = cipher_hash.hex()
        print(f"    Cipher file path : {cipher_path}")
        print(f"    Cipher length    : {len(ciphertext)} bytes")
        print(f"    Cipher SHA-256   : {cipher_hash_hex}")
        print(f"    Cipher hex head  : {format_hex_head(ciphertext, 32)}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to encrypt: {e}")
        sys.exit(1)
    
    # 5) DECRYPT
    print("[5] Reloading ciphertext from disk")
    try:
        with open(cipher_path, "rb") as f:
            ciphertext2 = f.read()
        cipher2_hash = hashlib.sha256(ciphertext2).digest()
        cipher2_hash_hex = cipher2_hash.hex()
        print(f"    Reloaded length  : {len(ciphertext2)} bytes")
        print(f"    SHA-256 (reload) : {cipher2_hash_hex}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to reload ciphertext: {e}")
        sys.exit(1)
    
    print("[6] Decrypting ciphertext -> plaintext")
    try:
        keystream2 = derive_keystream_from_features(features, len(ciphertext2))
        decrypted = decrypt_bytes(ciphertext2, keystream2)
        
        decrypted_hash = hashlib.sha256(decrypted).digest()
        decrypted_hash_hex = decrypted_hash.hex()
        print(f"    Decrypted length : {len(decrypted)} bytes")
        print(f"    Decrypted SHA-256: {decrypted_hash_hex}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to decrypt: {e}")
        sys.exit(1)
    
    # 6) COMPARE ORIGINAL VS DECRYPTED
    print("[7] Comparing original vs decrypted")
    if plaintext_bytes == decrypted:
        print("=" * 50)
        print("RESULT: ✅ ROUND-TRIP MATCH")
        print("=" * 50)
        print()
        
        if not raw:
            try:
                decrypted_text = decrypted.decode("utf-8", errors="replace")
                print("[8] Decrypted text preview:")
                print(f"    {safe_preview_text(decrypted_text, 200)}")
                print()
            except Exception as e:
                print(f"WARNING: Could not decode decrypted text: {e}")
    else:
        print("=" * 50)
        print("RESULT: ❌ ROUND-TRIP MISMATCH")
        print("=" * 50)
        print()
        
        # Show first 32 bytes of both
        print("First 32 bytes comparison:")
        print(f"Original:  {format_hex_head(plaintext_bytes, 32)}")
        print(f"Decrypted: {format_hex_head(decrypted, 32)}")
        
        # Show differences
        print("\nFirst 10 differing positions:")
        diff_count = 0
        for i, (orig_byte, dec_byte) in enumerate(zip(plaintext_bytes, decrypted)):
            if orig_byte != dec_byte:
                print(f"  Position {i}: orig={orig_byte:02x}, dec={dec_byte:02x}")
                diff_count += 1
                if diff_count >= 10:
                    break
        
        if diff_count == 0:
            print("  (Lengths differ)")
            if len(plaintext_bytes) != len(decrypted):
                print(f"  Original length: {len(plaintext_bytes)}")
                print(f"  Decrypted length: {len(decrypted)}")
        
        sys.exit(1)
    
    # 7) WRITE DECRYPTED OUTPUT FILE
    print("[8] Writing decrypted output file")
    try:
        if not raw:
            decrypted_text = decrypted.decode("utf-8", errors="replace")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(decrypted_text)
        else:
            with open(output_path, "wb") as f:
                f.write(decrypted)
        
        print(f"    Path: {output_path}")
        print()
        print("✅ Round-trip test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        sys.exit(1)


def main() -> None:
    """Parse CLI arguments and run the round-trip test."""
    parser = argparse.ArgumentParser(
        description="WorldSound OTP v2 Debug Round-Trip Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ws_debug_roundtrip.py \\
    --world world.wav \\
    --input message.txt \\
    --cipher cipher_output.bin \\
    --output decrypted_output.txt

  python ws_debug_roundtrip.py \\
    --world world.wav \\
    --input payload.bin \\
    --cipher cipher_output.bin \\
    --output decrypted_payload.bin \\
    --raw
        """
    )
    
    parser.add_argument(
        "--world", "-w",
        required=True,
        help="Path to world audio file (WAV) used as key material"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to plaintext text file (UTF-8)"
    )
    
    parser.add_argument(
        "--cipher", "-c",
        default="cipher_output.bin",
        help="Path to save ciphertext bytes (default: cipher_output.bin)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="decrypted_output.txt",
        help="Path to save decrypted text (default: decrypted_output.txt)"
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat input file as raw bytes instead of UTF-8 text (still print a safe preview)"
    )
    
    args = parser.parse_args()
    
    run_roundtrip(
        world_audio_path=args.world,
        input_path=args.input,
        cipher_path=args.cipher,
        output_path=args.output,
        raw=args.raw
    )


if __name__ == "__main__":
    main()
