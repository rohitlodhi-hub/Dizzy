import numpy as np
import wave

def generate_alert_sound(filename="alert.wav", duration=0.5, frequency=1000):
    """
    Generate a simple beep sound and save as WAV file
    
    Args:
        filename: Output filename
        duration: Sound duration in seconds
        frequency: Beep frequency in Hz
    """
    sample_rate = 44100
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    tone = np.sin(frequency * 2 * np.pi * t)
    
    # Add envelope to avoid clicks
    envelope = np.ones(samples)
    fade_samples = int(sample_rate * 0.01)  # 10ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    tone = tone * envelope
    
    # Convert to 16-bit PCM
    audio = np.int16(tone * 32767)
    
    # Save as WAV
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print(f"✓ Alert sound created: {filename}")

if __name__ == "__main__":
    generate_alert_sound("alert.wav")