from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import pandas as pd
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.05)
            detections += detections_batch       
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        tracks={
            "players":[],
            "referees":[],
            "goalkeeper":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            

            
            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert GoalKeeper to player object
            # for object_ind , class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            


            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["goalkeeper"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['goalkeeper']:
                    tracks["goalkeeper"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks       
 
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        circle_radius = 15  # Уменьшенный радиус круга
        x1_circle = x_center
        y1_circle = y2 + 15  # Смещаем круг вниз

        if track_id is not None:
            # Рисуем круг
            cv2.circle(
                frame,
                (int(x1_circle), int(y1_circle)),  # Центр круга
                int(circle_radius),  # Радиус круга
                color,
                cv2.FILLED
            )
            
            # Получаем размер текста
            text = f"{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Вычисляем позицию текста, чтобы он был по центру круга
            text_x = int(x1_circle - text_size[0] / 2)
            text_y = int(y1_circle + text_size[1] / 2)
            
            # Рисуем текст
            cv2.putText(
                frame,
                text,
                (text_x, text_y),  # Центрируем текст
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
            
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Параметры прямоугольника
        rect_start = (1350, 850)
        rect_end = (1900, 970)
        rect_color = (200, 100, 150)  # Интересный цвет
        alpha = 0.6  # Прозрачность
        corner_radius = 20  # Радиус закругленных углов

        # Создаем копию кадра для наложения
        overlay = frame.copy()
        
        # Рисуем закругленный прямоугольник
        cv2.rectangle(overlay, (rect_start[0] + corner_radius, rect_start[1]), (rect_end[0] - corner_radius, rect_end[1]), rect_color, -1)
        cv2.rectangle(overlay, (rect_start[0], rect_start[1] + corner_radius), (rect_end[0], rect_end[1] - corner_radius), rect_color, -1)
        cv2.circle(overlay, (rect_start[0] + corner_radius, rect_start[1] + corner_radius), corner_radius, rect_color, -1)
        cv2.circle(overlay, (rect_end[0] - corner_radius, rect_start[1] + corner_radius), corner_radius, rect_color, -1)
        cv2.circle(overlay, (rect_start[0] + corner_radius, rect_end[1] - corner_radius), corner_radius, rect_color, -1)
        cv2.circle(overlay, (rect_end[0] - corner_radius, rect_end[1] - corner_radius), corner_radius, rect_color, -1)

        # Накладываем полупрозрачный прямоугольник на кадр
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Вычисляем контроль мяча командами
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Избегаем деления на ноль
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = team_2 = 0

        # Отображаем текст контроля мяча
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame


    
    def draw_annotations(self,video_frames, tracks, team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeeper"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color")
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))
            
            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw Goalkeeper
            for _, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper["bbox"],(255,0,0))   
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
            
            
        return output_video_frames 
     