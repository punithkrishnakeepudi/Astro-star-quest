import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import pygame
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Initialize Pygame for audio
pygame.mixer.init()

class GameState(Enum):
    MENU = 1
    STORY = 2
    CALIBRATION = 3
    COUNTDOWN = 4
    PLAYING = 5
    GAME_OVER = 6
    LEVEL_COMPLETE = 7
    VICTORY = 8

@dataclass
class Star:
    x: float
    y: float
    collected: bool = False
    radius: int = 15
    pulse: float = 0.0
    glow: float = 0.0

@dataclass
class Enemy:
    x: float
    y: float
    speed: float
    radius: int = 25
    type: str = "alien"
    pulse: float = 0.0
    spawn_time: float = 0.0

@dataclass
class Gate:
    x: float
    y: float
    width: int = 80
    height: int = 100
    glow: float = 0.0

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple

class AstronautStarQuest:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Get screen dimensions and setup camera
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Create fullscreen window
        cv2.namedWindow('Astronaut Star Quest', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Astronaut Star Quest', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Game state
        self.state = GameState.MENU
        self.level = 1
        self.score = 0
        self.stars_collected = 0
        self.stars_needed = 65
        self.lives = 3

        # Player position
        self.player_x = self.screen_width // 2
        self.player_y = self.screen_height // 2
        self.player_size = 40

        # Hand tracking
        self.hand_detected = False
        self.hand_missing_time = 0
        self.hand_missing_alert = False
        self.calibration_time = 0
        self.countdown_time = 0

        # Game objects
        self.stars: List[Star] = []
        self.enemies: List[Enemy] = []
        self.gate: Optional[Gate] = None
        self.particles: List[Particle] = []

        # Background elements
        self.background_stars = []
        self.generate_background_stars()

        # Enemy spawn control
        self.last_enemy_spawn = 0
        self.enemy_spawn_interval = 2.0  # seconds between spawns

        # Animation timers
        self.animation_time = 0
        self.story_time = 0
        self.story_phase = 0
        self.alert_time = 0

        # Space nebula effect
        self.nebula_offset = 0

        # Audio setup
        self.setup_audio()

        # Level progression
        self.level_zones = [
            {"name": "Violet Plains", "levels": range(1, 6), "bg_color": (100, 50, 150)},
            {"name": "Crater Canyons", "levels": range(6, 11), "bg_color": (80, 40, 120)},
            {"name": "Sky Fragment Fields", "levels": range(11, 16), "bg_color": (60, 80, 140)},
            {"name": "Shattered Moonscape", "levels": range(16, 21), "bg_color": (40, 60, 100)},
            {"name": "Alaine Warzone", "levels": range(21, 25), "bg_color": (120, 40, 40)},
            {"name": "Gate of Solara", "levels": [25], "bg_color": (200, 180, 100)}
        ]

        self.generate_level()

    def setup_audio(self):
        """Setup audio effects"""
        try:
            pygame.mixer.set_num_channels(8)
            self.collect_sound = self.generate_tone(800, 0.1)
            self.collision_sound = self.generate_noise(0.2)
        except Exception as e:
            print(f"Audio setup failed: {e}")
            self.collect_sound = None
            self.collision_sound = None

    def generate_tone(self, frequency, duration):
        """Generate a simple tone"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            for i in range(frames):
                wave = np.sin(2 * np.pi * frequency * i / sample_rate)
                arr[i] = [wave * 0.3, wave * 0.3]
            sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            return sound
        except:
            return None

    def generate_noise(self, duration):
        """Generate noise for collision sound"""
        try:
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.random.random((frames, 2)) * 0.2 - 0.1
            sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            return sound
        except:
            return None

    def generate_background_stars(self):
        """Generate background stars for space effect"""
        self.background_stars = []
        for _ in range(200):
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)
            size = random.randint(1, 3)
            brightness = random.uniform(0.3, 1.0)
            twinkle_speed = random.uniform(0.5, 2.0)
            self.background_stars.append({
                'x': x, 'y': y, 'size': size,
                'brightness': brightness, 'twinkle_speed': twinkle_speed,
                'twinkle_phase': random.uniform(0, math.pi * 2)
            })

    def get_zone_info(self, level):
        for zone in self.level_zones:
            if level in zone["levels"]:
                return zone
        return self.level_zones[0]

    def generate_level(self):
        """Generate stars and enemies for current level"""
        self.stars = []
        self.enemies = []
        self.gate = None
        self.stars_collected = 0
        self.particles = []
        for i in range(self.stars_needed):
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            star = Star(x, y)
            star.pulse = random.uniform(0, math.pi * 2)
            star.glow = random.uniform(0, math.pi * 2)
            self.stars.append(star)

    def create_particles(self, x, y, color, count=10):
        """Create particle effect"""
        for _ in range(count):
            vx = random.uniform(-50, 50)
            vy = random.uniform(-50, 50)
            life = random.uniform(0.5, 1.5)
            self.particles.append(Particle(x, y, vx, vy, life, life, color))

    def update_particles(self, dt):
        """Update particle system"""
        for particle in self.particles[:]:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.life -= dt
            particle.vy += 50 * dt
            if particle.life <= 0:
                self.particles.remove(particle)

    def draw_space_background(self, overlay):
        """Draw animated space background"""
        self.nebula_offset += 0.5
        for star in self.background_stars:
            star['twinkle_phase'] += star['twinkle_speed'] * 0.02
            brightness = star['brightness'] * (0.5 + 0.5 * math.sin(star['twinkle_phase']))
            color_val = int(255 * brightness)
            color = (color_val, color_val, color_val)
            size = max(1, star['size'])
            cv2.circle(overlay, (star['x'], star['y']), size, color, -1)
        for i in range(50):
            x = int((i * 137 + self.nebula_offset * 10) % self.screen_width)
            y = int((i * 113 + self.nebula_offset * 3) % self.screen_height)
            alpha = math.sin(self.animation_time * 2 + i) * 0.3 + 0.5
            color_val = int(100 * alpha)
            cv2.circle(overlay, (x, y), 1, (color_val, color_val, color_val + 50), -1)

    def draw_story_animation(self, frame):
        """Draw the opening story animation"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (5, 5, 20)
        self.draw_space_background(overlay)
        story_texts = [
            "Year 3042: Earth is gone...",
            "Commander Ryn crash-lands on a violet moon",
            "Navigate 25 hostile zones",
            "Collect 65 stars per level",
            "Avoid RED enemies (-15 points)",
            "Reach the Gate of Solara",
            "Humanity's last hope..."
        ]
        phase_duration = 3.0
        self.story_phase = int(self.story_time / phase_duration)
        if self.story_phase < len(story_texts):
            text = story_texts[self.story_phase]
            phase_progress = (self.story_time % phase_duration) / phase_duration
            alpha = phase_progress / 0.2 if phase_progress < 0.2 else (1.0 - phase_progress) / 0.2 if phase_progress > 0.8 else 1.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = self.screen_height // 2
            glow_color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
            cv2.putText(overlay, text, (text_x + 2, text_y + 2), font, 2, glow_color, 8, cv2.LINE_AA)
            cv2.putText(overlay, text, (text_x, text_y), font, 2, (int(255 * alpha), int(255 * alpha), int(255 * alpha)), 3, cv2.LINE_AA)
        if self.story_time > len(story_texts) * phase_duration:
            self.state = GameState.CALIBRATION
            self.calibration_time = 0
        return overlay

    def draw_calibration(self, frame):
        """Draw calibration screen"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 10, 40)
        self.draw_space_background(overlay)
        pulse = math.sin(self.calibration_time * 3) * 0.3 + 0.7
        center = (self.screen_width // 2, self.screen_height // 2)
        radius = max(0, int(150 * pulse))
        cv2.circle(overlay, center, radius, (0, 255, 255), 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Calibrating Hand Tracking..."
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(overlay, text, (text_x, self.screen_height // 2 - 200), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        text2 = "Show your RIGHT hand to the camera"
        text2_size = cv2.getTextSize(text2, font, 1.5, 2)[0]
        text2_x = (self.screen_width - text2_size[0]) // 2
        cv2.putText(overlay, text2, (text2_x, self.screen_height // 2 + 200), font, 1.5, (200, 200, 200), 2, cv2.LINE_AA)
        progress = min(self.calibration_time / 5.0, 1.0)
        bar_width = int(600 * progress)
        bar_x = (self.screen_width - 600) // 2
        bar_y = self.screen_height // 2 + 100
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 600, bar_y + 20), (255, 255, 255), 3)
        if self.calibration_time > 5.0:
            self.state = GameState.COUNTDOWN
            self.countdown_time = 0
        return overlay

    def draw_countdown(self, frame):
        """Draw countdown screen"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (5, 5, 30)
        self.draw_space_background(overlay)
        countdown_num = 3 - int(self.countdown_time)
        if countdown_num > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(countdown_num)
            text_size = cv2.getTextSize(text, font, 15, 15)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = (self.screen_height + text_size[1]) // 2
            pulse = math.sin(self.countdown_time * 10) * 0.3 + 0.7
            color = (int(255 * pulse), int(255 * pulse), 255)
            cv2.putText(overlay, text, (text_x + 5, text_y + 5), font, 15, (50, 50, 100), 20, cv2.LINE_AA)
            cv2.putText(overlay, text, (text_x, text_y), font, 15, color, 15, cv2.LINE_AA)
        elif countdown_num == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "GO!"
            text_size = cv2.getTextSize(text, font, 12, 12)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = (self.screen_height + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x + 5, text_y + 5), font, 12, (0, 100, 0), 15, cv2.LINE_AA)
            cv2.putText(overlay, text, (text_x, text_y), font, 12, (0, 255, 0), 12, cv2.LINE_AA)
        if self.countdown_time > 4.0:
            self.state = GameState.PLAYING
        return overlay

    def draw_menu(self, frame):
        """Draw main menu"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 10, 30)
        self.draw_space_background(overlay)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "ASTRONAUT STAR QUEST"
        title_size = cv2.getTextSize(title, font, 4, 5)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 3
        glow_intensity = int(100 + 50 * math.sin(self.animation_time * 2))
        cv2.putText(overlay, title, (title_x + 3, title_y + 3), font, 4, (0, 0, glow_intensity), 8, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
        options = ["Press SPACE to Start", "Press S for Story", "Press Q to Quit"]
        for i, option in enumerate(options):
            y = title_y + 150 + i * 80
            option_size = cv2.getTextSize(option, font, 2, 3)[0]
            option_x = (self.screen_width - option_size[0]) // 2
            cv2.putText(overlay, option, (option_x, y), font, 2, (200, 200, 200), 3, cv2.LINE_AA)
        return overlay

    def draw_game(self, frame):
        """Draw main game"""
        zone = self.get_zone_info(self.level)
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        base_color = tuple(int(c * 0.3) for c in zone["bg_color"])
        overlay[:] = base_color
        self.draw_space_background(overlay)
        for particle in self.particles:
            alpha = particle.life / particle.max_life
            color = tuple(int(c * alpha) for c in particle.color)
            cv2.circle(overlay, (int(particle.x), int(particle.y)), 3, color, -1)
        for star in self.stars:
            if not star.collected:
                star.pulse += 0.08
                star.glow += 0.12
                pulse_size = max(0, int(star.radius + math.sin(star.pulse) * 8))
                glow_intensity = math.sin(star.glow) * 0.5 + 0.5
                glow_color = (int(100 * glow_intensity), int(150 * glow_intensity), int(255 * glow_intensity))
                cv2.circle(overlay, (int(star.x), int(star.y)), pulse_size + 10, glow_color, -1)
                cv2.circle(overlay, (int(star.x), int(star.y)), pulse_size, (255, 255, 100), -1)
                cv2.circle(overlay, (int(star.x), int(star.y)), max(0, pulse_size - 8), (255, 255, 255), -1)
        for enemy in self.enemies:
            enemy.pulse += 0.15
            pulse_scale = 1.0 + math.sin(enemy.pulse) * 0.4
            current_radius = max(0, int(enemy.radius * pulse_scale))
            glow_intensity = math.sin(enemy.pulse * 2) * 0.5 + 0.5
            glow_color = (0, 0, int(255 * glow_intensity))
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), current_radius + 15, glow_color, -1)
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), current_radius, (0, 0, 255), -1)
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), max(0, current_radius - 8), (255, 100, 100), -1)
        if self.gate:
            self.gate.glow += 0.2
            glow_intensity = int(150 + math.sin(self.gate.glow) * 100)
            cv2.rectangle(overlay, (int(self.gate.x - 20), int(self.gate.y - 20)), (int(self.gate.x + self.gate.width + 20), int(self.gate.y + self.gate.height + 20)), (glow_intensity // 2, glow_intensity // 2, glow_intensity), -1)
            cv2.rectangle(overlay, (int(self.gate.x), int(self.gate.y)), (int(self.gate.x + self.gate.width), int(self.gate.y + self.gate.height)), (glow_intensity, glow_intensity, 255), -1)
        player_glow = int(100 + 50 * math.sin(self.animation_time * 4))
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), self.player_size + 10, (0, player_glow, player_glow), -1)
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), self.player_size, (255, 255, 255), -1)
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), max(0, self.player_size - 15), (100, 200, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        hud_size = 1.5
        line_height = 60
        cv2.putText(overlay, f"Level: {self.level}", (30, 50), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Score: {self.score}", (30, 50 + line_height), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Stars: {self.stars_collected}/{self.stars_needed}", (30, 50 + 2 * line_height), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Zone: {zone['name']}", (30, 50 + 3 * line_height), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        if not self.hand_detected:
            self.alert_time += 0.02
            alert_alpha = math.sin(self.alert_time * 10) * 0.5 + 0.5
            alert_color = (0, 0, int(255 * alert_alpha))
            cv2.rectangle(overlay, (self.screen_width - 400, 30), (self.screen_width - 30, 120), alert_color, -1)
            cv2.rectangle(overlay, (self.screen_width - 400, 30), (self.screen_width - 30, 120), (255, 255, 255), 3)
            cv2.putText(overlay, "HAND MISSING!", (self.screen_width - 380, 80), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, "Hand Detected", (self.screen_width - 300, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

    def update_game_logic(self):
        """Update game physics and logic"""
        if self.state != GameState.PLAYING:
            return
        dt = 1 / 30
        current_time = time.time()
        self.update_particles(dt)
        if current_time - self.last_enemy_spawn > self.enemy_spawn_interval:
            self.spawn_enemy()
            self.last_enemy_spawn = current_time
        for enemy in self.enemies[:]:
            enemy.x -= enemy.speed
            if enemy.x < -100:
                self.enemies.remove(enemy)
                continue
            dx = enemy.x - self.player_x
            dy = enemy.y - self.player_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < (enemy.radius + self.player_size):
                self.score = max(0, self.score - 15)
                self.create_particles(enemy.x, enemy.y, (255, 0, 0), 15)
                if self.collision_sound:
                    self.collision_sound.play()
                self.enemies.remove(enemy)
        for star in self.stars:
            if not star.collected:
                dx = star.x - self.player_x
                dy = star.y - self.player_y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < (star.radius + self.player_size):
                    star.collected = True
                    self.stars_collected += 1
                    self.score += 10
                    self.create_particles(star.x, star.y, (255, 255, 0), 8)
                    if self.collect_sound:
                        self.collect_sound.play()
        if self.stars_collected >= self.stars_needed and not self.gate:
            self.gate = Gate(self.screen_width - 200, self.screen_height // 2 - 50)
        if self.gate:
            if (self.gate.x < self.player_x < self.gate.x + self.gate.width and
                    self.gate.y < self.player_y < self.gate.y + self.gate.height):
                if self.level >= 25:
                    self.state = GameState.VICTORY
                else:
                    self.level += 1
                    self.state = GameState.LEVEL_COMPLETE
                    self.generate_level()

    def spawn_enemy(self):
        """Spawn a single enemy"""
        y = random.randint(50, self.screen_height - 50)
        speed = 3 + (self.level - 1) * 0.4
        enemy = Enemy(self.screen_width + 50, y, speed, type="red_danger")
        enemy.spawn_time = time.time()
        self.enemies.append(enemy)

    def process_hand_tracking(self, frame):
        """Process hand tracking and update player position"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            self.hand_detected = True
            self.hand_missing_time = 0
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                finger_x = int(index_tip.x * w)
                finger_y = int(index_tip.y * h)
                target_x = int((finger_x / w) * self.screen_width)
                target_y = int((finger_y / h) * self.screen_height)
                self.player_x += (target_x - self.player_x) * 0.2
                self.player_y += (target_y - self.player_y) * 0.2
                self.player_x = max(self.player_size, min(self.screen_width - self.player_size, self.player_x))
                self.player_y = max(self.player_size, min(self.screen_height - self.player_size, self.player_y))
        else:
            self.hand_detected = False
            self.hand_missing_time += 1 / 30

    def handle_input(self, key):
        """Handle keyboard input"""
        if self.state == GameState.MENU:
            if key == ord(' '):
                self.state = GameState.STORY
                self.story_time = 0
            elif key == ord('s') or key == ord('S'):
                self.state = GameState.STORY
                self.story_time = 0
            elif key == ord('q') or key == ord('Q'):
                return False
        elif self.state == GameState.GAME_OVER:
            if key == ord('r') or key == ord('R'):
                self.level = 1
                self.score = 0
                self.generate_level()
                self.state = GameState.PLAYING
            elif key == ord('m') or key == ord('M'):
                self.state = GameState.MENU
        elif self.state == GameState.LEVEL_COMPLETE:
            if key == ord(' '):
                self.state = GameState.PLAYING
        elif self.state == GameState.VICTORY:
            if key == ord(' '):
                self.state = GameState.MENU
                self.level = 1
                self.score = 0
                self.generate_level()
        return True

    def draw_game_over(self, frame):
        """Draw game over screen"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (30, 10, 10)
        self.draw_space_background(overlay)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "GAME OVER"
        title_size = cv2.getTextSize(title, font, 5, 6)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 100
        cv2.putText(overlay, title, (title_x + 5, title_y + 5), font, 5, (100, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 5, (255, 255, 255), 6, cv2.LINE_AA)
        score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(score_text, font, 2.5, 3)[0]
        score_x = (self.screen_width - score_size[0]) // 2
        cv2.putText(overlay, score_text, (score_x, title_y + 120), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        restart_text = "Press R to Restart"
        restart_size = cv2.getTextSize(restart_text, font, 2, 2)[0]
        restart_x = (self.screen_width - restart_size[0]) // 2
        cv2.putText(overlay, restart_text, (restart_x, title_y + 200), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        menu_text = "Press M for Menu"
        menu_size = cv2.getTextSize(menu_text, font, 2, 2)[0]
        menu_x = (self.screen_width - menu_size[0]) // 2
        cv2.putText(overlay, menu_text, (menu_x, title_y + 250), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def draw_level_complete(self, frame):
        """Draw level complete screen"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 30, 10)
        self.draw_space_background(overlay)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "LEVEL COMPLETE!"
        title_size = cv2.getTextSize(title, font, 4, 5)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 50
        cv2.putText(overlay, title, (title_x + 3, title_y + 3), font, 4, (0, 100, 0), 8, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
        level_text = f"Level {self.level - 1} Cleared!"
        level_size = cv2.getTextSize(level_text, font, 2.5, 3)[0]
        level_x = (self.screen_width - level_size[0]) // 2
        cv2.putText(overlay, level_text, (level_x, title_y + 100), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        continue_text = "Press SPACE to Continue"
        continue_size = cv2.getTextSize(continue_text, font, 2, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        cv2.putText(overlay, continue_text, (continue_x, title_y + 180), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def draw_victory(self, frame):
        """Draw victory screen"""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (50, 50, 10)
        self.draw_space_background(overlay)
        font = cv2.FONT_HERSHEY_SIMPLEX
        glow_intensity = int(150 + 100 * math.sin(self.animation_time * 3))
        title = "VICTORY!"
        title_size = cv2.getTextSize(title, font, 6, 8)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 150
        cv2.putText(overlay, title, (title_x + 5, title_y + 5), font, 6, (glow_intensity // 2, glow_intensity // 2, 0), 12, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 6, (255, 255, 255), 8, cv2.LINE_AA)
        messages = ["Gate of Solara Reached!", "Humanity is Saved!", f"Final Score: {self.score}"]
        for i, message in enumerate(messages):
            msg_size = cv2.getTextSize(message, font, 2.5, 3)[0]
            msg_x = (self.screen_width - msg_size[0]) // 2
            msg_y = title_y + 120 + i * 70
            cv2.putText(overlay, message, (msg_x, msg_y), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        menu_text = "Press SPACE for Menu"
        menu_size = cv2.getTextSize(menu_text, font, 2, 2)[0]
        menu_x = (self.screen_width - menu_size[0]) // 2
        cv2.putText(overlay, menu_text, (menu_x, title_y + 350), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def run(self):
        """Main game loop"""
        clock = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            dt = current_time - clock
            clock = current_time
            self.animation_time += dt
            self.process_hand_tracking(frame)
            if self.state == GameState.STORY:
                self.story_time += dt
            elif self.state == GameState.CALIBRATION:
                self.calibration_time += dt
            elif self.state == GameState.COUNTDOWN:
                self.countdown_time += dt
            self.update_game_logic()
            display_frame = None
            if self.state == GameState.MENU:
                display_frame = self.draw_menu(frame)
            elif self.state == GameState.STORY:
                display_frame = self.draw_story_animation(frame)
            elif self.state == GameState.CALIBRATION:
                display_frame = self.draw_calibration(frame)
            elif self.state == GameState.COUNTDOWN:
                display_frame = self.draw_countdown(frame)
            elif self.state == GameState.PLAYING:
                display_frame = self.draw_game(frame)
            elif self.state == GameState.GAME_OVER:
                display_frame = self.draw_game_over(frame)
            elif self.state == GameState.LEVEL_COMPLETE:
                display_frame = self.draw_level_complete(frame)
            elif self.state == GameState.VICTORY:
                display_frame = self.draw_victory(frame)
            if display_frame is not None:
                cv2.imshow('Astronaut Star Quest', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key != 255:
                if not self.handle_input(key):
                    break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        game = AstronautStarQuest()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required libraries installed:")
        print("pip install opencv-python mediapipe numpy pygame")
    finally:
        cv2.destroyAllWindows()