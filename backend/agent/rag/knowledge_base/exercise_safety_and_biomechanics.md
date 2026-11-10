# General Exercise Safety, Warm-Up Protocols and 2D Pose Estimation Principles
**Source:** ACSM Guidelines for Exercise Testing and Prescription (11th Edition) & MediaPipe Pose Architecture (Lugaresi et al., 2019)
**Citation:** American College of Sports Medicine. (2021). ACSM's guidelines for exercise testing and prescription. Wolters Kluwer.

## 1. Warm-Up Protocols and Exercise Safety
- **General Warm-Up:** 5 to 10 minutes of light dynamic cardiovascular activity (jogging in place, jumping jacks) to raise core body temperature and increase tissue elasticity.
- **Dynamic Mobility:** Focus on target joint mobility:
  - Ankle dorsiflexion and hip openers (world's greatest stretch, deep goblet squat holds) before squats and lunges.
  - Thoracic spine rotations, cat-camels, and band pull-aparts before shoulder presses and push-ups.
- **Warm-Up Sets:** Perform 1 to 2 sets with 40-50% target weight before working sets to groove motor pathways.

## 2. Fundamentals of 2D Vision-Based Joint Angle Measurement
- **Vector Cosine Principle:** Given joint coordinates $A(x_1, y_1)$, $B(x_2, y_2)$ (vertex), and $C(x_3, y_3)$, the angle $\theta$ at vertex $B$ is computed via:
  $$\vec{u} = A - B, \quad \vec{v} = C - B$$
  $$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}, \quad \theta = \arccos(\text{clamp}(\cos(\theta), -1.0, 1.0))$$
- **Camera Perspective Invariance & Parallax:** 2D angles represent projections of 3D kinematic movements onto the camera sensor plane. When movement occurs oblique to the sensor, projected angles may foreshorten. This underscores the necessity of the pre-detector Camera View Suitability Module to enforce appropriate viewpoint alignment.
- **Hysteresis in Repetition Counting:** Finite-state machines must incorporate hysteresis bands (e.g. 155° overhead threshold vs 95° bottom threshold) to prevent rapid state chatter or double counting around boundary transitions.

