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

# Initialize Pygame's mixer for audio effects
pygame.mixer.init()

# Define game states using an Enum for clarity and type safety
class GameState(Enum):
    MENU = 1          # Main menu screen
    STORY = 2         # Story introduction sequence
    CALIBRATION = 3   # Hand tracking calibration phase
    COUNTDOWN = 4     # Countdown before gameplay starts
    PLAYING = 5       # Active gameplay state
    GAME_OVER = 6     # Game over screen
    LEVEL_COMPLETE = 7 # Level completion screen
    VICTORY = 8       # Final victory screen

# Data class to represent collectible stars
@dataclass
class Star:
    x: float          # X-coordinate of the star
    y: float          # Y-coordinate of the star
    collected: bool = False  # Whether the star has been collected
    radius: int = 15         # Radius of the star
    pulse: float = 0.0       # Animation pulse for visual effect
    glow: float = 0.0        # Glow animation phase

# Data class to represent enemy objects
@dataclass
class Enemy:
    x: float          # X-coordinate of the enemy
    y: float          # Y-coordinate of the enemy
    speed: float      # Movement speed of the enemy
    radius: int = 25  # Radius of the enemy
    type: str = "alien"  # Type of enemy (e.g., "red_danger")
    pulse: float = 0.0   # Animation pulse for visual effect
    spawn_time: float = 0.0  # Time when the enemy was spawned

# Data class to represent the level gate
@dataclass
class Gate:
    x: float          # X-coordinate of the gate
    y: float          # Y-coordinate of the gate
    width: int = 80   # Width of the gate
    height: int = 100 # Height of the gate
    glow: float = 0.0 # Glow animation phase

# Data class to represent particle effects
@dataclass
class Particle:
    x: float          # X-coordinate of the particle
    y: float          # Y-coordinate of the particle
    vx: float         # X-velocity of the particle
    vy: float         # Y-velocity of the particle
    life: float       # Current lifespan of the particle
    max_life: float   # Maximum lifespan of the particle
    color: tuple      # RGB color of the particle

class AstronautStarQuest:
    def __init__(self):
        """Initialize the game, setting up MediaPipe, camera, game state, and assets."""
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # Continuous tracking mode
            max_num_hands=1,               # Track only one hand
            min_detection_confidence=0.7,   # Minimum confidence for hand detection
            min_tracking_confidence=0.5     # Minimum confidence for tracking
        )

        # Get screen dimensions using Tkinter
        import tkinter as tk
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()   # Screen width in pixels
        self.screen_height = root.winfo_screenheight() # Screen height in pixels
        root.destroy()

        # Set up webcam capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Set webcam resolution to 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Create a fullscreen OpenCV window
        cv2.namedWindow('Astronaut Star Quest', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Astronaut Star Quest', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Initialize game state variables
        self.state = GameState.MENU           # Start in the menu state
        self.level = 1                        # Current level (starts at 1)
        self.score = 0                        # Player's score
        self.stars_collected = 0              # Number of stars collected in current level
        self.stars_needed = 65                # Stars required to complete a level
        self.lives = 3                        # Player's remaining lives

        # Initialize player position at screen center
        self.player_x = self.screen_width // 2
        self.player_y = self.screen_height // 2
        self.player_size = 40                 # Size of the player avatar

        # Initialize hand tracking variables
        self.hand_detected = False            # Whether a hand is currently detected
        self.hand_missing_time = 0            # Time since hand was last detected
        self.hand_missing_alert = False       # Alert state for missing hand
        self.calibration_time = 0             # Timer for calibration phase
        self.countdown_time = 0               # Timer for countdown phase

        # Initialize game objects
        self.stars: List[Star] = []           # List of stars in the current level
        self.enemies: List[Enemy] = []        # List of enemies in the current level
        self.gate: Optional[Gate] = None      # Gate object (appears after collecting all stars)
        self.particles: List[Particle] = []   # List of active particle effects

        # Initialize background elements
        self.background_stars = []            # List of background stars for visual effect
        self.generate_background_stars()       # Populate background stars

        # Initialize enemy spawn control
        self.last_enemy_spawn = 0             # Time of last enemy spawn
        self.enemy_spawn_interval = 2.0       # Seconds between enemy spawns

        # Initialize animation timers
        self.animation_time = 0               # Global animation timer
        self.story_time = 0                   # Timer for story sequence
        self.story_phase = 0                  # Current phase of story animation
        self.alert_time = 0                   # Timer for hand missing alert

        # Initialize nebula effect offset
        self.nebula_offset = 0                # Offset for nebula animation

        # Set up audio effects
        self.setup_audio()

        # Define level zones with names, level ranges, and background colors
        self.level_zones = [
            {"name": "Violet Plains", "levels": range(1, 6), "bg_color": (100, 50, 150)},
            {"name": "Crater Canyons", "levels": range(6, 11), "bg_color": (80, 40, 120)},
            {"name": "Sky Fragment Fields", "levels": range(11, 16), "bg_color": (60, 80, 140)},
            {"name": "Shattered Moonscape", "levels": range(16, 21), "bg_color": (40, 60, 100)},
            {"name": "Alaine Warzone", "levels": range(21, 25), "bg_color": (120, 40, 40)},
            {"name": "Gate of Solara", "levels": [25], "bg_color": (200, 180, 100)}
        ]

        # Generate initial level
        self.generate_level()

    def setup_audio(self):
        """Configure audio effects for star collection and enemy collisions."""
        try:
            # Set up 8 audio channels for concurrent sound playback
            pygame.mixer.set_num_channels(8)
            # Generate sound for collecting stars
            self.collect_sound = self.generate_tone(800, 0.1)
            # Generate sound for enemy collisions
            self.collision_sound = self.generate_noise(0.2)
        except Exception as e:
            # Handle audio setup errors gracefully
            print(f"Audio setup failed: {e}")
            self.collect_sound = None
            self.collision_sound = None

    def generate_tone(self, frequency, duration):
        """Generate a sine wave tone for sound effects."""
        try:
            sample_rate = 22050  # Audio sample rate
            frames = int(duration * sample_rate)  # Number of audio frames
            arr = np.zeros((frames, 2))  # Stereo audio array
            for i in range(frames):
                # Generate sine wave for both channels
                wave = np.sin(2 * np.pi * frequency * i / sample_rate)
                arr[i] = [wave * 0.3, wave * 0.3]  # Scale amplitude
            # Convert array to Pygame sound
            sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            return sound
        except:
            return None

    def generate_noise(self, duration):
        """Generate random noise for collision sound effects."""
        try:
            sample_rate = 22050  # Audio sample rate
            frames = int(duration * sample_rate)  # Number of audio frames
            arr = np.random.random((frames, 2)) * 0.2 - 0.1  # Random noise scaled
            # Convert array to Pygame sound
            sound = pygame.sndarray.make_sound((arr * 32767).astype(np.int16))
            return sound
        except:
            return None

    def generate_background_stars(self):
        """Generate 200 background stars for a twinkling space effect."""
        self.background_stars = []
        for _ in range(200):
            # Randomly position stars within screen bounds
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)
            size = random.randint(1, 3)  # Random size between 1 and 3 pixels
            brightness = random.uniform(0.3, 1.0)  # Random brightness
            twinkle_speed = random.uniform(0.5, 2.0)  # Random twinkle speed
            # Add star with initial twinkle phase
            self.background_stars.append({
                'x': x, 'y': y, 'size': size,
                'brightness': brightness, 'twinkle_speed': twinkle_speed,
                'twinkle_phase': random.uniform(0, math.pi * 2)
            })

    def get_zone_info(self, level):
        """Retrieve the zone information for the given level."""
        for zone in self.level_zones:
            if level in zone["levels"]:
                return zone
        return self.level_zones[0]  # Default to first zone if level not found

    def generate_level(self):
        """Generate stars and reset game objects for the current level."""
        self.stars = []           # Clear existing stars
        self.enemies = []         # Clear existing enemies
        self.gate = None          # Reset gate
        self.stars_collected = 0  # Reset collected stars
        self.particles = []       # Clear particles
        # Generate required number of stars
        for _ in range(self.stars_needed):
            # Randomly position stars within screen bounds (with padding)
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            star = Star(x, y)
            # Randomize animation phases
            star.pulse = random.uniform(0, math.pi * 2)
            star.glow = random.uniform(0, math.pi * 2)
            self.stars.append(star)

    def create_particles(self, x, y, color, count=10):
        """Create particle effects at the given position with specified color."""
        for _ in range(count):
            # Randomize particle velocity and lifespan
            vx = random.uniform(-50, 50)
            vy = random.uniform(-50, 50)
            life = random.uniform(0.5, 1.5)
            self.particles.append(Particle(x, y, vx, vy, life, life, color))

    def update_particles(self, dt):
        """Update particle positions and remove expired particles."""
        for particle in self.particles[:]:
            # Update position based on velocity
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            # Apply gravity to vertical velocity
            particle.vy += 50 * dt
            # Decrease particle lifespan
            particle.life -= dt
            # Remove particle if its lifespan is over
            if particle.life <= 0:
                self.particles.remove(particle)

    def draw_space_background(self, overlay):
        """Draw a dynamic space background with twinkling stars and cosmic dust."""
        self.nebula_offset += 0.5  # Update nebula animation offset
        # Draw twinkling background stars
        for star in self.background_stars:
            star['twinkle_phase'] += star['twinkle_speed'] * 0.02
            brightness = star['brightness'] * (0.5 + 0.5 * math.sin(star['twinkle_phase']))
            color_val = int(255 * brightness)
            color = (color_val, color_val, color_val)
            size = max(1, star['size'])  # Ensure size is at least 1
            cv2.circle(overlay, (star['x'], star['y']), size, color, -1)
        # Draw moving cosmic dust particles
        for i in range(50):
            x = int((i * 137 + self.nebula_offset * 10) % self.screen_width)
            y = int((i * 113 + self.nebula_offset * 3) % self.screen_height)
            alpha = math.sin(self.animation_time * 2 + i) * 0.3 + 0.5
            color_val = int(100 * alpha)
            cv2.circle(overlay, (x, y), 1, (color_val, color_val, color_val + 50), -1)

    def draw_story_animation(self, frame):
        """Render the story introduction sequence with fading text."""
        # Create a blank overlay for the story screen
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (5, 5, 20)  # Set deep space background color
        self.draw_space_background(overlay)  # Draw background stars
        # Define story text phases
        story_texts = [
            "Year 3042: Earth is gone...",
            "Commander Ryn crash-lands on a violet moon",
            "Navigate 25 hostile zones",
            "Collect 65 stars per level",
            "Avoid RED enemies (-15 points)",
            "Reach the Gate of Solara",
            "Humanity's last hope..."
        ]
        phase_duration = 3.0  # Duration of each story phase
        self.story_phase = int(self.story_time / phase_duration)  # Current phase
        # Display text if within story phase count
        if self.story_phase < len(story_texts):
            text = story_texts[self.story_phase]
            # Calculate fade-in/out effect
            phase_progress = (self.story_time % phase_duration) / phase_duration
            alpha = phase_progress / 0.2 if phase_progress < 0.2 else (1.0 - phase_progress) / 0.2 if phase_progress > 0.8 else 1.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Calculate text position
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = self.screen_height // 2
            # Draw glowing text shadow
            glow_color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
            cv2.putText(overlay, text, (text_x + 2, text_y + 2), font, 2, glow_color, 8, cv2.LINE_AA)
            # Draw main text
            cv2.putText(overlay, text, (text_x, text_y), font, 2, (int(255 * alpha), int(255 * alpha), int(255 * alpha)), 3, cv2.LINE_AA)
        # Transition to calibration after story completes
        if self.story_time > len(story_texts) * phase_duration:
            self.state = GameState.CALIBRATION
            self.calibration_time = 0
        return overlay

    def draw_calibration(self, frame):
        """Render the hand tracking calibration screen."""
        # Create a blank overlay for the calibration screen
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 10, 40)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        # Calculate pulsing effect for calibration circle
        pulse = math.sin(self.calibration_time * 3) * 0.3 + 0.7
        center = (self.screen_width // 2, self.screen_height // 2)
        radius = max(0, int(150 * pulse))  # Ensure non-negative radius
        # Draw calibration circle
        cv2.circle(overlay, center, radius, (0, 255, 255), 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw calibration instruction text
        text = "Calibrating Hand Tracking..."
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(overlay, text, (text_x, self.screen_height // 2 - 200), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        # Draw hand instruction text
        text2 = "Show your RIGHT hand to the camera"
        text2_size = cv2.getTextSize(text2, font, 1.5, 2)[0]
        text2_x = (self.screen_width - text2_size[0]) // 2
        cv2.putText(overlay, text2, (text2_x, self.screen_height // 2 + 200), font, 1.5, (200, 200, 200), 2, cv2.LINE_AA)
        # Draw progress bar
        progress = min(self.calibration_time / 5.0, 1.0)
        bar_width = int(600 * progress)
        bar_x = (self.screen_width - 600) // 2
        bar_y = self.screen_height // 2 + 100
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + 600, bar_y + 20), (255, 255, 255), 3)
        # Transition to countdown after 5 seconds
        if self.calibration_time > 5.0:
            self.state = GameState.COUNTDOWN
            self.countdown_time = 0
        return overlay

    def draw_countdown(self, frame):
        """Render the countdown screen before gameplay starts."""
        # Create a blank overlay for the countdown screen
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (5, 5, 30)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        countdown_num = 3 - int(self.countdown_time)  # Calculate current countdown number
        if countdown_num > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(countdown_num)
            text_size = cv2.getTextSize(text, font, 15, 15)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = (self.screen_height + text_size[1]) // 2
            # Apply pulsing effect to text
            pulse = math.sin(self.countdown_time * 10) * 0.3 + 0.7
            color = (int(255 * pulse), int(255 * pulse), 255)
            # Draw glowing text shadow
            cv2.putText(overlay, text, (text_x + 5, text_y + 5), font, 15, (50, 50, 100), 20, cv2.LINE_AA)
            # Draw main countdown text
            cv2.putText(overlay, text, (text_x, text_y), font, 15, color, 15, cv2.LINE_AA)
        elif countdown_num == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "GO!"
            text_size = cv2.getTextSize(text, font, 12, 12)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            text_y = (self.screen_height + text_size[1]) // 2
            # Draw glowing "GO!" text
            cv2.putText(overlay, text, (text_x + 5, text_y + 5), font, 12, (0, 100, 0), 15, cv2.LINE_AA)
            cv2.putText(overlay, text, (text_x, text_y), font, 12, (0, 255, 0), 12, cv2.LINE_AA)
        # Transition to playing state after 4 seconds
        if self.countdown_time > 4.0:
            self.state = GameState.PLAYING
        return overlay

    def draw_menu(self, frame):
        """Render the main menu screen with title and options."""
        # Create a blank overlay for the menu
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 10, 30)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw game title
        title = "ASTRONAUT STAR QUEST"
        title_size = cv2.getTextSize(title, font, 4, 5)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 3
        # Apply glowing effect to title
        glow_intensity = int(100 + 50 * math.sin(self.animation_time * 2))
        cv2.putText(overlay, title, (title_x + 3, title_y + 3), font, 4, (0, 0, glow_intensity), 8, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
        # Draw menu options
        options = ["Press SPACE to Start", "Press S for Story", "Press Q to Quit"]
        for i, option in enumerate(options):
            y = title_y + 150 + i * 80
            option_size = cv2.getTextSize(option, font, 2, 3)[0]
            option_x = (self.screen_width - option_size[0]) // 2
            cv2.putText(overlay, option, (option_x, y), font, 2, (200, 200, 200), 3, cv2.LINE_AA)
        return overlay

    def draw_game(self, frame):
        """Render the main gameplay screen with stars, enemies, gate, and HUD."""
        # Get current zone information
        zone = self.get_zone_info(self.level)
        # Create a blank overlay for the game
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        base_color = tuple(int(c * 0.3) for c in zone["bg_color"])  # Dimmed zone color
        overlay[:] = base_color
        self.draw_space_background(overlay)  # Draw background stars
        # Draw particles
        for particle in self.particles:
            alpha = particle.life / particle.max_life  # Fade based on lifespan
            color = tuple(int(c * alpha) for c in particle.color)
            cv2.circle(overlay, (int(particle.x), int(particle.y)), 3, color, -1)
        # Draw uncollected stars
        for star in self.stars:
            if not star.collected:
                star.pulse += 0.08  # Update pulse animation
                star.glow += 0.12   # Update glow animation
                pulse_size = max(0, int(star.radius + math.sin(star.pulse) * 8))  # Ensure non-negative radius
                glow_intensity = math.sin(star.glow) * 0.5 + 0.5
                # Draw outer glow
                glow_color = (int(100 * glow_intensity), int(150 * glow_intensity), int(255 * glow_intensity))
                cv2.circle(overlay, (int(star.x), int(star.y)), pulse_size + 10, glow_color, -1)
                # Draw main star
                cv2.circle(overlay, (int(star.x), int(star.y)), pulse_size, (255, 255, 100), -1)
                # Draw inner star
                cv2.circle(overlay, (int(star.x), int(star.y)), max(0, pulse_size - 8), (255, 255, 255), -1)
        # Draw enemies
        for enemy in self.enemies:
            enemy.pulse += 0.15  # Update pulse animation
            pulse_scale = 1.0 + math.sin(enemy.pulse) * 0.4
            current_radius = max(0, int(enemy.radius * pulse_scale))  # Ensure non-negative radius
            glow_intensity = math.sin(enemy.pulse * 2) * 0.5 + 0.5
            # Draw outer danger glow
            glow_color = (0, 0, int(255 * glow_intensity))
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), current_radius + 15, glow_color, -1)
            # Draw main enemy body
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), current_radius, (0, 0, 255), -1)
            # Draw inner enemy detail
            cv2.circle(overlay, (int(enemy.x), int(enemy.y)), max(0, current_radius - 8), (255, 100, 100), -1)
        # Draw gate if present
        if self.gate:
            self.gate.glow += 0.2  # Update gate glow animation
            glow_intensity = int(150 + math.sin(self.gate.glow) * 100)
            # Draw outer gate glow
            cv2.rectangle(overlay, (int(self.gate.x - 20), int(self.gate.y - 20)),
                          (int(self.gate.x + self.gate.width + 20), int(self.gate.y + self.gate.height + 20)),
                          (glow_intensity // 2, glow_intensity // 2, glow_intensity), -1)
            # Draw main gate
            cv2.rectangle(overlay, (int(self.gate.x), int(self.gate.y)),
                          (int(self.gate.x + self.gate.width), int(self.gate.y + self.gate.height)),
                          (glow_intensity, glow_intensity, 255), -1)
        # Draw player with glow effect
        player_glow = int(100 + 50 * math.sin(self.animation_time * 4))
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), self.player_size + 10,
                   (0, player_glow, player_glow), -1)
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), self.player_size, (255, 255, 255), -1)
        cv2.circle(overlay, (int(self.player_x), int(self.player_y)), max(0, self.player_size - 15),
                   (100, 200, 255), -1)
        # Draw HUD elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        hud_size = 1.5
        line_height = 60
        cv2.putText(overlay, f"Level: {self.level}", (30, 50), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Score: {self.score}", (30, 50 + line_height), font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Stars: {self.stars_collected}/{self.stars_needed}", (30, 50 + 2 * line_height),
                    font, hud_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"Zone: {zone['name']}", (30, 50 + 3 * line_height), font, hud_size, (255, 255, 255), 3,
                    cv2.LINE_AA)
        # Display hand tracking status
        if not self.hand_detected:
            self.alert_time += 0.02  # Update alert animation
            alert_alpha = math.sin(self.alert_time * 10) * 0.5 + 0.5
            alert_color = (0, 0, int(255 * alert_alpha))
            # Draw alert box for missing hand
            cv2.rectangle(overlay, (self.screen_width - 400, 30), (self.screen_width - 30, 120), alert_color, -1)
            cv2.rectangle(overlay, (self.screen_width - 400, 30), (self.screen_width - 30, 120), (255, 255, 255), 3)
            cv2.putText(overlay, "HAND MISSING!", (self.screen_width - 380, 80), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Indicate successful hand detection
            cv2.putText(overlay, "Hand Detected", (self.screen_width - 300, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

    def update_game_logic(self):
        """Update game physics, collisions, and state transitions during gameplay."""
        if self.state != GameState.PLAYING:
            return
        dt = 1 / 30  # Fixed time step (assuming 30 FPS)
        current_time = time.time()
        # Update particle system
        self.update_particles(dt)
        # Spawn enemies at regular intervals
        if current_time - self.last_enemy_spawn > self.enemy_spawn_interval:
            self.spawn_enemy()
            self.last_enemy_spawn = current_time
        # Update and check enemy collisions
        for enemy in self.enemies[:]:
            enemy.x -= enemy.speed  # Move enemy left
            if enemy.x < -100:  # Remove enemies off-screen
                self.enemies.remove(enemy)
                continue
            # Check collision with player
            dx = enemy.x - self.player_x
            dy = enemy.y - self.player_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < (enemy.radius + self.player_size):
                self.score = max(0, self.score - 15)  # Deduct score, prevent negative
                self.create_particles(enemy.x, enemy.y, (255, 0, 0), 15)  # Red particles
                if self.collision_sound:
                    self.collision_sound.play()  # Play collision sound
                self.enemies.remove(enemy)
        # Check star collisions
        for star in self.stars:
            if not star.collected:
                dx = star.x - self.player_x
                dy = star.y - self.player_y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < (star.radius + self.player_size):
                    star.collected = True
                    self.stars_collected += 1
                    self.score += 10  # Add score for star
                    self.create_particles(star.x, star.y, (255, 255, 0), 8)  # Yellow particles
                    if self.collect_sound:
                        self.collect_sound.play()  # Play collect sound
        # Spawn gate when all stars are collected
        if self.stars_collected >= self.stars_needed and not self.gate:
            self.gate = Gate(self.screen_width - 200, self.screen_height // 2 - 50)
        # Check gate collision
        if self.gate:
            if (self.gate.x < self.player_x < self.gate.x + self.gate.width and
                    self.gate.y < self.player_y < self.gate.y + self.gate.height):
                if self.level >= 25:
                    self.state = GameState.VICTORY  # Win game at level 25
                else:
                    self.level += 1
                    self.state = GameState.LEVEL_COMPLETE  # Advance to next level
                    self.generate_level()

    def spawn_enemy(self):
        """Spawn a new enemy on the right side of the screen."""
        y = random.randint(50, self.screen_height - 50)  # Random Y position
        speed = 3 + (self.level - 1) * 0.4  # Increase speed with level
        enemy = Enemy(self.screen_width + 50, y, speed, type="red_danger")
        enemy.spawn_time = time.time()
        self.enemies.append(enemy)

    def process_hand_tracking(self, frame):
        """Process webcam frame to track hand and update player position."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            self.hand_detected = True
            self.hand_missing_time = 0
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip landmark
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                # Convert to screen coordinates
                finger_x = int(index_tip.x * w)
                finger_y = int(index_tip.y * h)
                target_x = int((finger_x / w) * self.screen_width)
                target_y = int((finger_y / h) * self.screen_height)
                # Smoothly interpolate player position
                self.player_x += (target_x - self.player_x) * 0.2
                self.player_y += (target_y - self.player_y) * 0.2
                # Keep player within screen bounds
                self.player_x = max(self.player_size, min(self.screen_width - self.player_size, self.player_x))
                self.player_y = max(self.player_size, min(self.screen_height - self.player_size, self.player_y))
        else:
            self.hand_detected = False
            self.hand_missing_time += 1 / 30  # Increment missing time

    def handle_input(self, key):
        """Handle keyboard inputs based on the current game state."""
        if self.state == GameState.MENU:
            if key == ord(' '):
                self.state = GameState.STORY
                self.story_time = 0
            elif key == ord('s') or key == ord('S'):
                self.state = GameState.STORY
                self.story_time = 0
            elif key == ord('q') or key == ord('Q'):
                return False  # Quit game
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
        """Render the game over screen with score and options."""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (30, 10, 10)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw "Game Over" title
        title = "GAME OVER"
        title_size = cv2.getTextSize(title, font, 5, 6)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 100
        cv2.putText(overlay, title, (title_x + 5, title_y + 5), font, 5, (100, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 5, (255, 255, 255), 6, cv2.LINE_AA)
        # Draw final score
        score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(score_text, font, 2.5, 3)[0]
        score_x = (self.screen_width - score_size[0]) // 2
        cv2.putText(overlay, score_text, (score_x, title_y + 120), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        # Draw restart option
        restart_text = "Press R to Restart"
        restart_size = cv2.getTextSize(restart_text, font, 2, 2)[0]
        restart_x = (self.screen_width - restart_size[0]) // 2
        cv2.putText(overlay, restart_text, (restart_x, title_y + 200), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        # Draw menu option
        menu_text = "Press M for Menu"
        menu_size = cv2.getTextSize(menu_text, font, 2, 2)[0]
        menu_x = (self.screen_width - menu_size[0]) // 2
        cv2.putText(overlay, menu_text, (menu_x, title_y + 250), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def draw_level_complete(self, frame):
        """Render the level completion screen."""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (10, 30, 10)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw "Level Complete" title
        title = "LEVEL COMPLETE!"
        title_size = cv2.getTextSize(title, font, 4, 5)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 50
        cv2.putText(overlay, title, (title_x + 3, title_y + 3), font, 4, (0, 100, 0), 8, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
        # Draw level cleared message
        level_text = f"Level {self.level - 1} Cleared!"
        level_size = cv2.getTextSize(level_text, font, 2.5, 3)[0]
        level_x = (self.screen_width - level_size[0]) // 2
        cv2.putText(overlay, level_text, (level_x, title_y + 100), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        # Draw continue instruction
        continue_text = "Press SPACE to Continue"
        continue_size = cv2.getTextSize(continue_text, font, 2, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        cv2.putText(overlay, continue_text, (continue_x, title_y + 180), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def draw_victory(self, frame):
        """Render the victory screen after completing all levels."""
        overlay = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        overlay[:] = (50, 50, 10)  # Set background color
        self.draw_space_background(overlay)  # Draw background stars
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw "Victory" title with glow
        glow_intensity = int(150 + 100 * math.sin(self.animation_time * 3))
        title = "VICTORY!"
        title_size = cv2.getTextSize(title, font, 6, 8)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = self.screen_height // 2 - 150
        cv2.putText(overlay, title, (title_x + 5, title_y + 5), font, 6,
                    (glow_intensity // 2, glow_intensity // 2, 0), 12, cv2.LINE_AA)
        cv2.putText(overlay, title, (title_x, title_y), font, 6, (255, 255, 255), 8, cv2.LINE_AA)
        # Draw victory messages
        messages = ["Gate of Solara Reached!", "Humanity is Saved!", f"Final Score: {self.score}"]
        for i, message in enumerate(messages):
            msg_size = cv2.getTextSize(message, font, 2.5, 3)[0]
            msg_x = (self.screen_width - msg_size[0]) // 2
            msg_y = title_y + 120 + i * 70
            cv2.putText(overlay, message, (msg_x, msg_y), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
        # Draw menu instruction
        menu_text = "Press SPACE for Menu"
        menu_size = cv2.getTextSize(menu_text, font, 2, 2)[0]
        menu_x = (self.screen_width - menu_size[0]) // 2
        cv2.putText(overlay, menu_text, (menu_x, title_y + 350), font, 2, (200, 200, 200), 2, cv2.LINE_AA)
        return overlay

    def run(self):
        """Run the main game loop, handling input, updates, and rendering."""
        clock = time.time()  # Initialize clock for delta time
        while True:
            # Capture webcam frame
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
            # Calculate delta time
            current_time = time.time()
            dt = current_time - clock
            clock = current_time
            self.animation_time += dt  # Update animation timer
            # Process hand tracking
            self.process_hand_tracking(frame)
            # Update state-specific timers
            if self.state == GameState.STORY:
                self.story_time += dt
            elif self.state == GameState.CALIBRATION:
                self.calibration_time += dt
            elif self.state == GameState.COUNTDOWN:
                self.countdown_time += dt
            # Update game logic
            self.update_game_logic()
            # Initialize display frame
            display_frame = None
            # Render appropriate screen based on game state
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
            # Display the frame if it exists
            if display_frame is not None:
                cv2.imshow('Astronaut Star Quest', display_frame)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key != 255:  # Ignore no-key press
                if not self.handle_input(key):
                    break
        # Cleanup resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create and run the game
        game = AstronautStarQuest()
        game.run()
    except Exception as e:
        # Handle and report any errors
        print(f"Error: {e}")
        print("Make sure you have the required libraries installed:")
        print("pip install opencv-python mediapipe numpy pygame")
    finally:
        # Ensure windows are closed
        cv2.destroyAllWindows()
