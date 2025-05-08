from ultralytics import YOLO
import supervision as sv
import pickle #savies  and loading tracking results
import os
import numpy as np
import pandas as pd
import cv2 #for drawing
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        # loops through all  players, referees, ball
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):  
                for track_id, track_info in track.items():  
                    bbox = track_info['bbox']
                    # for ball we use center, for others we use foot position
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position  # store position in dict

    def interpolate_ball_positions(self, ball_positions):
        # extract the ball bbox from each frame (track id is always 1 for ball)
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()  # fill missing using linear interpolation
        df_ball_positions = df_ball_positions.bfill()  # backfill if interpolation can't fill at the start

        # convert DataFrame back to list of dicts in original format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20  # process 20 frames at a time for efficiency
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)  # run YOLO prediction
            detections += detections_batch  # collect all detections
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # if already processed before and stub is available, just load saved result
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)  # get YOLO detections for all frames

        # initialize empty structure for storing tracking results
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # change from   0: 'player',  to  'player': 0 as supervision.Detections use ids
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # map of class_id to name
            cls_names_inv = {v: k for k, v in cls_names.items()}  # map of name to class_id

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # prepare empty containers for each type
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # store tracked player and referee boxes
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # bounding box
                cls_id = frame_detection[3]  # class
                track_id = frame_detection[4]  # tracking ID

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # separately store ball bbox using raw detections
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # fixed ID for ball

        # optionally save tracking results to file
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # bottom y of the bbox
        x_center, _ = get_center_of_bbox(bbox)  # x center of the bbox
        width = get_bbox_width(bbox)  # width of the box

        # draw ellipse shape at the foot of the player
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45, #circle start   
            endAngle=235, #circle end
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # draw player ID box 
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2 # calculate the left x-coordinate of the rectangle by centering it around x_center
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15 # calculate the top y-coordinate of the rectangle (a bit above the player’s foot)
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            # draw rectangle behind player ID
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            # adjust text position if ID is longer than 2 digits
            x1_text = x1_rect + 12 #add padding 12 pixels
            if track_id > 99:
                x1_text -= 10

            # ID text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, #font size
                (0, 0, 0), #color black
                2 #thickness
            )

        return frame
    #the arrow on top of the ball
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])  # top y of bbox
        x, _ = get_center_of_bbox(bbox)  # x center of bbox

        # triangle tip and base points
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        # draw filled triangle with outline
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2) #border of trianle

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4  # transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # slice array up to current frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # count how many frames each team had the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # draw the text on screen
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        # loop over all frames
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # avoid modifying original frame

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw each player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # fallback color is red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # if player has the ball, draw a triangle
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # draw  referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0)) #green color

            # overlay team ball control stats
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
