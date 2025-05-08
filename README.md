# AI-Powered Football Match Analysis

## Abstract/Introduction

This project focuses on leveraging state-of-the-art Artificial Intelligence techniques, primarily object detection, tracking, clustering, and optical flow, to perform an in-depth analysis of football match videos. Starting with basic object detection using pre-trained YOLO models, the project progressively refines its approach by fine-tuning models with custom datasets, implementing robust object tracking, assigning players to teams based on jersey colors, determining ball possession, calculating team ball control statistics, and estimating camera movement. The ultimate goal is to extract meaningful insights from video footage, such as player movements, team formations, and possession dynamics, with custom, clear visualizations.

## 1. Problem Statement

Standard pre-trained object detection models, while powerful, exhibit several limitations when applied directly to complex football match scenarios. These include:
* **Inaccurate/Inconsistent Object Detection:** Particularly for fast-moving small objects like the football.
* **Lack of Specificity:** Generic models detect all "persons" without distinguishing between players, referees, coaching staff, or audience members, which is crucial for focused game analysis.
* **No Role Differentiation:** Inability to differentiate roles like players and referees automatically.
* **Temporal Incoherence:** Frame-by-frame detection alone does not provide persistent identities for objects, making it difficult to track individual player actions or ball trajectory over time.
* **Ball Possession Ambiguity:** Without further logic, it's unclear which player controls the ball.
* **Impact of Camera Motion:** Player movements can be confounded by camera movements, leading to inaccurate analysis of speed and distance covered if not accounted for.
* **Obtrusive Visualizations:** Default annotations from detection libraries can be visually cluttered and obscure important game action.

This project aims to address these challenges by developing a specialized AI pipeline for football video analysis.

## 2. Objectives

The primary objectives of this project are:

* **Develop Accurate Object Detection:** To reliably detect key entities in a football match: players, referees, goalkeepers, and the ball.
* **Implement Robust Object Tracking:** To assign and maintain unique identifiers for each player and the referee throughout the video.
* **Differentiate and Filter Entities:** To focus analysis only on active participants on the field and distinguish between player roles.
* **Automated Team Assignment:** To automatically assign players to their respective teams based on visual cues (t-shirt color).
* **Continuous Ball Tracking:** To ensure the ball's position is known for every frame, interpolating its path during brief periods of missed detection.
* **Assign Ball Possession:** To determine which player has possession of the ball based on proximity.
* **Calculate Team Ball Control:** To compute and display the percentage of time each team possesses the ball.
* **Estimate Camera Movement:** To measure camera pan and tilt across frames using optical flow on static scene parts.
* **Custom Visualizations:** To create clear, informative, and non-obtrusive annotations on the video output.
* **Structured and Reusable Codebase:** To develop a modular project structure for clarity and future extensions.

**Future/Planned Objectives:**
* Compensate player movement using the estimated camera movement.
* Apply perspective transformation to enable measurements in real-world units (e.g., meters).
* Calculate advanced metrics like true player speed and total distance covered.
* Advanced Tactical Analysis: Explore detection of formations, player heatmaps.

## 3. Methodology

The project is developed in phases, building upon previous results to achieve a comprehensive analysis pipeline.

### 3.1. Data Acquisition and Preparation

* **Input Video:**
    * Source: Kaggle dataset "dfl Bundesliga data shootout."
    * Specific File: `08Fd334.mp4` (featuring an "Eagle Eye" camera perspective).
    * Storage: `input_videos/` subfolder within the project directory.
* **Custom Training Dataset (for Fine-tuning):**
    * Source: Roboflow "Football Player Detection image dataset."
    * Annotations: "player," "referee," "goalkeeper," "ball."
    * Size: 612 images (train, test, valid splits).
    * Format: YOLOv5 format.
    * Preparation: Downloaded via Roboflow API, directory structure adjusted using `shutil`. `data.yaml` configures dataset paths and class names.

### 3.2. Core AI Models and Techniques

#### 3.2.1. Object Detection
* **Initial Exploration:** YOLOv8x (`yolov8x.pt`) via Ultralytics.
* **Fine-tuning for Football Specifics:**
    * Base Model: YOLOv5x (`yolov5x.pt`).
    * Training: Google Colab, 100 epochs, image size 640x640.
    * Output: `models/best.pt`.

#### 3.2.2. Object Tracking
* **Algorithm:** ByteTrack, via `supervision` library (`sv`).
* **Implementation (`trackers/tracker.py` - `Tracker` class):**
    * Uses fine-tuned YOLOv5 model for detections.
    * Goalkeepers re-labeled as "player" pre-tracking.
    * Tracked data (`tracks`) structured per frame for players, referees, ball with IDs and bounding boxes.
    * Caching: Tracks saved/loaded via `pickle` (`stubs/track_stubs.pkl`).

#### 3.2.3. Color-based Team Assignment
* **Algorithm:** K-Means Clustering (`sklearn.cluster.KMeans`).
* **Implementation (`team_assigner/team_assigner.py` - `TeamAssigner` class):**
    * Extracts player t-shirt color (K-Means on top half of player crop).
    * Identifies two main team colors from the first frame's player shirt colors (K-Means on extracted shirt colors). `n_init` for K-Means increased to 10 for robustness.
    * Assigns players to Team 1 or Team 2.
    * **Refinement:** Hardcoded fix for known goalkeeper's team assignment if initial color-based assignment is incorrect.
    * Results cached and stored in `tracks`.

#### 3.2.4. Ball Position Interpolation
* **Technique:** Linear Interpolation using `pandas`.
* **Implementation (`Tracker.interpolate_ball_positions`):**
    * Converts ball bounding box data to Pandas DataFrame.
    * Uses `DataFrame.interpolate()` and `DataFrame.backfill()`.
    * Updates `tracks['ball']`.

#### 3.2.5. Player-Ball Possession Assignment
* **Module:** `player_ball_assigner/player_ball_assigner.py` (`PlayerBallAssigner` class).
* **Logic (`assign_ball_to_player`):**
    * Calculates Euclidean distance from ball center (`utils.bbox_utils.get_center_of_bounding_box`) to player's inferred foot positions (bottom-left `(x1, y2)` and bottom-right `(x2, y2)` of player's bounding box) using `utils.bbox_utils.measure_distance`.
    * The player whose closer foot is within `max_player_ball_distance` (e.g., 70 pixels) and is the minimum among all players is assigned possession.
    * Updates `tracks['players'][frame_num][player_id]` with `has_ball: True`.

#### 3.2.6. Team Ball Control Percentage Calculation
* **Implementation (primarily in `main.py` and visualization in `Tracker` class):**
    * **Data Collection:** A `team_ball_control` list in `main.py` stores the team ID of the player in possession for each frame. If no player has the ball, the last possessing team's ID is carried over.
    * **Display (`Tracker.draw_team_ball_control`):**
        * Draws a semi-transparent background rectangle on the frame.
        * Calculates percentage of frames each team had control up to the current frame.
        * Uses `cv2.putText` to display these statistics (e.g., "Team 1 Ball Control: XX.XX%").

#### 3.2.7. Camera Movement Estimation
* **Module:** `camera_movement_estimator/camera_movement_estimator.py` (`CameraMovementEstimator` class).
* **Technique:** Lucas-Kanade Optical Flow.
* **Logic (`get_camera_movement`):**
    * Feature Detection: `cv2.goodFeaturesToTrack` detects corners in masked regions (e.g., top/bottom static banners) of grayscale frames. Parameters: `maxCorners=100`, `qualityLevel=0.3`, `minDistance=3`.
    * Optical Flow: `cv2.calcOpticalFlowPyrLK` tracks these features from the previous frame (`old_gray`) to the current frame (`frame_gray`). LK parameters: `winSize=(15,15)`, `maxLevel=2`.
    * Movement Calculation: The (dx, dy) movement is determined by the feature pair exhibiting the maximum Euclidean displacement between frames. If this `max_distance_this_frame` exceeds a `minimum_distance` threshold (e.g., 5 pixels), the movement `[dx, dy]` (calculated using `utils.bbox_utils.measure_xy_distance` as `old_pos - new_pos`) is recorded for the frame.
    * Feature Re-detection: `old_features` are re-detected on each `frame_gray` to track to the subsequent frame, rather than tracking initial features throughout the video.
    * Caching: Camera movements (`camera_movement_per_frame`) saved/loaded via `pickle` (`stubs/camera_movement.pkl`).
    * Visualization: `draw_camera_movement` method is defined for future visualization of camera vectors.

### 3.3. Software, Libraries, and Tools
* **Python**
* **AI/ML:** `ultralytics`, `roboflow`, `supervision`, `scikit-learn`, `pandas`.
* **Video/Image:** `OpenCV-Python` (`cv2`).
* **Development:** Local machine, Jupyter Notebooks, Google Colab.
* **Utilities:** `numpy`, `shutil`, `pickle`, `os`, `sys`, `matplotlib.pyplot`.

### 3.4. Project Workflow and Structure
(Simplified flow focusing on additions)
1.  ... (Initial setup, video read, object tracking) ...
2.  **Ball Interpolation:** `tracker.interpolate_ball_positions`.
3.  **Team Assignment:** `TeamAssigner` assigns teams.
4.  **Ball Possession & Team Control Data:**
    * Loop through frames:
        * `player_ball_assigner.assign_ball_to_player` determines player with ball.
        * Update `tracks` with `has_ball` status.
        * Append possessing team's ID to `team_ball_control` list.
5.  **Camera Movement Estimation:** `camera_movement_estimator.get_camera_movement`.
6.  **Custom Annotation & Display:**
    * `tracker.draw_annotations` (includes player ellipses, IDs, ball triangle, and possession indicator).
    * `tracker.draw_team_ball_control` (displays team stats).
    * (Future: `camera_movement_estimator.draw_camera_movement`).
7.  **Video Output:** `utils.video_utils.save_video`.

**Key Utility Functions (in `utils/bbox_utils.py`):**
* `get_center_of_bounding_box(bbox)`
* `get_bbox_width(bbox)`
* `measure_distance(p1, p2)`: Euclidean distance.
* `measure_xy_distance(p1, p2)`: Returns `(p1[0]-p2[0], p1[1]-p2[1])`.

**Directory Structure additions:**
football-analysis/
├── ... (previous folders)
├── player_ball_assigner/
│   ├── init.py
│   └── player_ball_assigner.py
├── camera_movement_estimator/
│   ├── init.py
│   └── camera_movement_estimator.py
├── stubs/
│   ├── ... (track_stubs.pkl)
│   └── camera_movement.pkl
└── ... (main.py, etc.)
### 3.5. Custom Annotations and Visualization
* **Player/Referee Annotation (`Tracker.draw_ellipse`):** As before (team-colored ellipses, player IDs).
* **Ball Annotation (`Tracker.draw_triangle`):** Green triangle for general ball position.
* **Possession Indicator:** If a player `has_ball`, an additional red triangle is drawn on their bounding box using `Tracker.draw_triangle(frame, player_bbox, color='red')`.
* **Team Ball Control Display:** Semi-transparent overlay showing real-time percentage stats for each team.

## 4. Results and Discussion
(Previous results for detection, fine-tuning, tracking, team assignment, ball interpolation remain relevant)

### 4.7. Ball Possession Assignment
* The system successfully assigns ball possession to the player whose inferred foot position is closest to the ball center and within a defined threshold (70 pixels).
* Visual verification via a red triangle on the possessing player confirms correct assignments in many instances.

### 4.8. Team Ball Control Statistics
* The team ball control percentages are calculated and displayed dynamically on the output video.
* The logic to carry over possession to the previously possessing team during brief moments of no explicit player possession (e.g., during a pass) ensures a continuous and more realistic representation of team control.
* The refinement in `TeamAssigner` (hardcoding goalkeeper's team and increasing K-Means `n_init`) improved the accuracy of the base team assignments, which in turn benefits the ball control statistics.

### 4.9. Camera Movement Estimation
* The `CameraMovementEstimator` successfully calculates a frame-by-frame (dx, dy) vector representing the dominant camera motion by tracking features in assumed static regions (banners).
* The use of feature re-detection per frame (rather than continuous tracking of initial features) aims to provide robust estimates even with longer sequences.
* The ability to cache these movements (`stubs/camera_movement.pkl`) is beneficial for development.
* Visual inspection of the (dx, dy) values or a (future) visualization would confirm if the estimated movement aligns with the perceived camera motion in the video.

## 5. Conclusion 

### 5.1. Conclusion
This project has successfully developed an AI pipeline for football match analysis, capable of robust object detection, tracking, role differentiation, team assignment, ball possession identification, team control statistics calculation, and camera movement estimation. Fine-tuning YOLO models, integrating ByteTrack, K-Means clustering for team assignment, and optical flow for camera analysis were key AI techniques employed. The system provides enriched video output with custom annotations and valuable statistics, laying a strong foundation for advanced football analytics.

## 6. References
(As previously listed, with the addition of implicit references to Optical Flow literature, e.g., Lucas-Kanade method).
* **Datasets:** Kaggle "dfl Bundesliga data shootout," Roboflow "Football Player Detection image dataset."
* **Models/Algorithms:** YOLO, ByteTrack, K-Means Clustering, Lucas-Kanade Optical Flow.
* **Libraries/Frameworks:** Ultralytics, Supervision, OpenCV-Python, Scikit-learn, Pandas, NumPy, Roboflow SDK.