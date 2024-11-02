<div align="center">

![Main](docs/soccer.gif)

## Overview

This project is a computer vision-based analysis tool for soccer matches. It leverages advanced image processing techniques and machine learning models to analyze and extract meaningful insights from soccer match videos, such as player tracking, ball movement, and event detection.

---

## 🛠 Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-0078D4?style=for-the-badge&logo=roboflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

---

## Player and Team Tracking

Detects players on the field and assigns each to their team color for easy identification. Referees are distinguished separately. Additionally, player numbers can be linked, ball tracking is active, and current ball possession is shown in real-time.

![Players](docs/giphy_1.gif)

</div>

## 🚀 How to Try?

1. Clone the repository:

```git clone <repository_url>```

2. Add an MP4 video file to the input_videos folder.
3. In the main file, specify the path to your input video by setting the file name accordingly.
4. Run the main file:
```python main.py```
5. The processed video will be saved in the output_videos folder.

1. **Video Input**: The service takes in soccer match videos as input.
   
   ![Video Input](docs/img/video_input.png)

2. **Object Detection & Tracking**: Using computer vision models (YOLO or similar), the service identifies and tracks players, referees, and the ball.

3. **Event Detection**: The system detects key events, such as goals, passes, tackles, and offsides.

4. **Data Export**: The extracted data (player movements, ball positions, and event timestamps) are stored in a database for further analysis.

5. **Visualization**: A visual representation of the analyzed data is generated, allowing users to review match dynamics in an intuitive format.

---

## 📦 Features

- **Player & Ball Tracking**: Tracks the movement of players and the ball in real-time.
- **Event Detection**: Automatically detects key game events like goals, offsides, and fouls.
- **Data Storage**: Saves match data, including positional information and events, in a PostgreSQL database for easy access.
- **Visualization**: Generates heatmaps and movement trajectories for players and the ball.
- **Customizable Detection Models**: You can fine-tune models or integrate your own models for more specific analysis.

---

## 🔧 Configuration

The service uses a YAML configuration file where you can:

- Specify the model for object detection (e.g., YOLOv5).
- Configure detection thresholds for different events.
- Set parameters for player tracking and ball tracking algorithms.
- Adjust frame rate and processing speeds to balance accuracy and performance.

---

## 📈 Logging

All detection and analysis tasks are logged, allowing users to trace back every step of the process for debugging or auditing purposes.

---

## 🤖 Customizable

The system can be customized and extended to support other sports or use cases by adjusting the configuration file or training new models. The flexibility of the architecture allows for rapid adaptation to different types of events or analysis tasks.

---

## 💻 For more detailed instructions, contact me on [Telegram](https://t.me/yourtelegram)

<p align="center">
  https://t.me/yourtelegram
</p>

