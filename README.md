
# 🚀 Astronaut Star Quest

A futuristic space adventure controlled by your hand movements!

---

## 🌌 Overview

**Astronaut Star Quest** is an immersive, gesture-controlled arcade game powered by computer vision. Step into the boots of **Commander Ryn**, humanity's last hope, as you navigate 25 perilous zones to collect stars, dodge hostile alien enemies, and reach the mythical **Gate of Solara** to save humankind.

- **Hand-tracking**: Powered by **MediaPipe** for seamless gesture control.  
- **Visuals**: Stunning space-themed graphics rendered with **OpenCV**.  
- **Audio**: Real-time sound effects using **Pygame**.  
- **Gameplay**: Fullscreen experience driven by **webcam input**.

---

## 🎮 Gameplay

- 👋 **Control**: Use your right hand’s index finger to guide Commander Ryn.  
- ✨ **Objective**: Collect **65 stars per level** to unlock the Gate.  
- 🚨 **Challenge**: Avoid red enemies — each collision deducts **15 points** from your score!  
- 🚪 **Progress**: Enter the Gate to advance through **25 unique levels**.  
- 🌟 **Zones**: Journey through visually distinct zones with increasing difficulty.

---

## 🕹️ Controls

### Hand Movement
- Move your **right hand** to control the astronaut.

### Keyboard
- **SPACE**: Start the game or continue from level completion/victory  
- **S**: View the story introduction  
- **R**: Restart after game over  
- **M**: Return to the main menu  
- **Q**: Quit the game  
- **ESC**: Exit the game at any time

---

## 📦 Requirements

To run the game, install the following Python packages:

```bash
pip install opencv-python mediapipe numpy pygame
```

### Prerequisites

- Python **3.7+**
- A working **webcam** for hand tracking
- A **well-lit environment** for accurate gesture detection

---

## 🛠️ How to Run

### Clone the repository:

```bash
git clone https://github.com/your-username/astro-star-quest.git
cd astro-star-quest
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the game:

```bash
python app.py
```

Ensure your webcam is connected and position your **right hand in view** to start playing!

---

## 🖼️ Screenshots

> _Add gameplay screenshots to the `assets/` folder and update the path below._

```markdown
![Gameplay Screenshot](assets/screenshot.png)
```

---

## 🎨 Zones

Explore six unique zones, each with distinct visuals and increasing challenges:

- **Violet Plains (Levels 1–5)**: A serene yet deceptive starting zone  
- **Crater Canyons (Levels 6–10)**: Rugged terrain with faster enemies  
- **Sky Fragment Fields (Levels 11–15)**: Chaotic asteroid-filled skies  
- **Shattered Moonscape (Levels 16–20)**: A desolate, high-risk region  
- **Alaine Warzone (Levels 21–24)**: Intense battles with relentless foes  
- **Gate of Solara (Level 25)**: The final challenge to save humanity

---

## 💡 Inspiration

Astronaut Star Quest combines **cutting-edge AI**, **computer vision**, and **retro arcade aesthetics** to create a unique, interactive gaming experience. It showcases the power of gesture-based controls, blending modern technology with classic arcade fun.

---

## 🔖 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- **MediaPipe**: For robust hand-tracking capabilities  
- **OpenCV**: For real-time computer vision and rendering  
- **Pygame**: For audio effects and game loop support  
- The **open-source community** for inspiration and resources

---

## 📁 Project Structure

```
astro-star-quest/
├── app.py             # Main game script
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License file
└── assets/            # Folder for images, sounds, etc.
    └── screenshot.png # Gameplay screenshot (placeholder)
```

---

## 📦 `requirements.txt`

```
opencv-python
mediapipe
numpy
pygame
```

---

## 🛠️ Troubleshooting

- **Webcam Issues**: Ensure your webcam is properly connected and not in use by other applications.  
- **Hand Tracking**: Play in a well-lit environment and keep your right hand clearly visible.  
- **Dependencies**: Verify all required libraries are installed correctly.  
- **Errors**: Check the console output for specific error messages and ensure Python 3.7+ is used.

For additional support, open an **issue** on the GitHub repository.

---

## 🌟 Contribute

Want to enhance **Astronaut Star Quest**?  
Contributions are welcome! Submit pull requests or suggest features via **GitHub Issues**.

---

> 🧑‍🚀 Embark on a cosmic journey and guide Commander Ryn to victory!
