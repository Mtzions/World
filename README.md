create new re
# WorldSound OTP - Version 2

WorldSound OTP v2 is a deterministic audio-based keystream generator and XOR-based encrypt/decrypt tool that uses multiple spectral features from an input audio file to derive a high-entropy keystream.

## Features

- **Deterministic**: Given the same audio file and parameters, the keystream will be identical on every run
- **Cryptographically Secure**: Uses SHA-256 hashing for key derivation
- **Audio-based**: Converts audio characteristics into cryptographic key material
- **Full Encryption/Decryption**: XOR-based encryption with proper round-trip functionality

## File-in-Sound v1 Integration

This implementation also includes File-in-Sound v1 functionality that combines WorldSound OTP v2 encryption with audio steganography:

1. **Encrypt any file** using WorldSound OTP v2
2. **Embed the encrypted data** into an audio file using steganography
3. **Extract and decrypt** the data from the stego audio

## How It Works

1. **Audio Analysis**: Extracts multiple spectral features from an audio file:
   - Dominant frequency index
   - Spectral centroid
   - Log-energy
   - Spectral flux
   - Phase difference

2. **Keystream Generation**: Uses SHA-256 hash of the features to derive a deterministic keystream

3. **Encryption/Decryption**: XORs the keystream with plaintext/ciphertext

## Installation

Requirements:
- Python 3.7+
- NumPy
- SciPy
- SoundFile
- Flask (for visualization)

Install dependencies:
```bash
pip install numpy scipy soundfile flask
```

## Usage

### Core WorldSound OTP v2 Functions

```bash
# Extract features from audio
python worldsound.py extract-key world.wav --hex

# Encrypt a file
python worldsound.py encrypt world.wav input.txt output.bin

# Decrypt a file
python worldsound.py decrypt world.wav input.bin output.txt
```

### File-in-Sound v1 Operations

```bash
# Encrypt and encode a file into audio
python worldsound.py encrypt-and-encode \
  --file secret.dex \
  --world world.wav \
  --cover music.wav \
  --out stego.wav

# Decode and decrypt a file from audio
python worldsound.py decode-and-decrypt \
  --stego stego.wav \
  --world world.wav \
  --out secret_restored.dex
```

### Help

```bash
python worldsound.py --help
python worldsound.py <subcommand> --help
```

## Core Functions

- `extract_features_from_audio(audio_path: str) -> bytes`
- `derive_keystream_from_features(features: bytes, length: int) -> bytes`
- `encrypt_bytes(plaintext: bytes, keystream: bytes) -> bytes`
- `decrypt_bytes(ciphertext: bytes, keystream: bytes) -> bytes`
- `encrypt_with_world_sound(audio_path: str, plaintext: bytes) -> bytes`
- `decrypt_with_world_sound(audio_path: str, ciphertext: bytes) -> bytes`

## File Format Specification

The File-in-Sound v1 payload format is:

```
MAGIC (4 bytes): b"WSF0"
LENGTH (4 bytes): uint32 big-endian (size of original file)
HASH (32 bytes): SHA-256 of the plaintext
CIPHER (N bytes): XOR-encrypted file content
```

## Requirements

- Python 3.6+
- numpy
- scipy
- soundfile

## Security Notes

- The system is deterministic: identical inputs always produce identical outputs
- No randomness is introduced beyond what's present in the audio
- The keystream is generated using a cryptographically secure hash function
- The system does not embed messages in audio - it only analyzes audio characteristics

## License

MIT
