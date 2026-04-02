
import os
import pygame

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SoundManager:

    def __init__(self, sound_file: str = "rasengan-sound-effect.mp3"):
        sound_path = os.path.join(_SCRIPT_DIR, sound_file)

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._enabled = True
        except Exception as e:
            print(f"[WARN] Sound init failed: {e}")
            self._enabled = False
            return

        # Load sound
        if os.path.isfile(sound_path):
            self._rasengan_sound = pygame.mixer.Sound(sound_path)
            self._rasengan_sound.set_volume(0.6)
            print(f"[INFO] Loaded sound: {sound_file}")
        else:
            print(f"[WARN] Sound file not found: {sound_path}")
            self._rasengan_sound = None
            self._enabled = False

        # State tracking
        self._rasengan_playing = False
        self._rasengan_channel = None

    def play_rasengan(self):
        """Start playing the Rasengan sound (loops until stopped)."""
        if not self._enabled or not self._rasengan_sound:
            return
        if not self._rasengan_playing:
            self._rasengan_channel = self._rasengan_sound.play(loops=-1)
            if self._rasengan_channel:
                self._rasengan_channel.set_volume(0.5)
            self._rasengan_playing = True

    def stop_rasengan(self, fade_ms: int = 300):
        """Stop the Rasengan loop sound with a fade-out."""
        if not self._enabled:
            return
        if self._rasengan_playing and self._rasengan_channel:
            self._rasengan_channel.fadeout(fade_ms)
            self._rasengan_playing = False

    def play_shoot(self):
        """Play a short Rasengan burst for the shoot action."""
        if not self._enabled or not self._rasengan_sound:
            return
        # Stop the loop first
        self.stop_rasengan(fade_ms=50)
        # Play once at higher volume on a separate channel
        channel = self._rasengan_sound.play(loops=0)
        if channel:
            channel.set_volume(0.8)

    @property
    def is_rasengan_playing(self) -> bool:
        return self._rasengan_playing

    def cleanup(self):
        """Shut down the mixer."""
        if self._enabled:
            pygame.mixer.quit()
