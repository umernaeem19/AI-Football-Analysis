from utils import read_video, save_video
from trackers import Tracker
import cv2  # OpenCV for image/video processing
import numpy as np
from team_assigner import TeamAssigner  # logic to assign players to teams based on color
from player_ball_assigner import PlayerBallAssigner  # handles which player currently has the ball
from camera_movement_estimator import CameraMovementEstimator  # estimates camera movement across frames
from view_transformer import ViewTransformer  # transforms field view to a top-down perspective
from speed_and_distance_estimator import SpeedAndDistance_Estimator  # estimates speed and distance of players and ball


def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')  

    # Initialize Tracker
    tracker = Tracker('models/best.pt')  # create tracker using YOLO model

    # run object detection and tracking, or load from a saved file
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Get object positions 
    tracker.add_position_to_tracks(tracks)  # calculate positions for all tracked objects (ball center or player foot)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])  # initialize movement estimator with first frame
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                               read_from_stub=True,
                                                                               stub_path='stubs/camera_movement_stub.pkl')  # estimate or load per-frame camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)  # shift positions to remove camera movement

    # View Transformer
    view_transformer = ViewTransformer()  # used to transform field view into top-down perspective
    view_transformer.add_transformed_position_to_tracks(tracks)  # apply the transformation to all tracked positions

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])  # smooth out missing or jittery ball detections

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()  # initialize class for calculating motion metrics
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)  # add speed and distance estimates to each player/ball per frame

    # #save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop box from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] y1 to y2 and x1 to x2

    #     #save cropped image
    #     cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)
    #     break


    # Assign Player Teams
    team_assigner = TeamAssigner()  # helper class to group players into teams based on jersey color
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])  # initialize team colors using the first frame’s player data
    
    # loop through all players in all frames to assign team identity and color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if frame_num < len(video_frames):
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)  # guess team from jersey color
            else:
                print(f"Warning: frame_num {frame_num} out of range (max index: {len(video_frames) - 1})")
            tracks['players'][frame_num][player_id]['team'] = team  # store team ID
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  # assign consistent team color for drawing

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()  # helper to figure out who has the ball in each frame
    team_ball_control = []  # keep track of which team has possession

    # loop through all frames and figure out who is closest to the ball
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # get ball bounding box for current frame
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)  # find the player closest to ball

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True  # mark that this player has the ball
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])  # record the team that has possession
        else:
            team_ball_control.append(team_ball_control[-1])  # if no one has ball, use last known team in control
    team_ball_control = np.array(team_ball_control)  # convert list to numpy array for easier indexing and processing later

    # DRAW OUTPUT
    ## draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)  # add bounding boxes, arrows, and team info to frames

    ## draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)  # show camera movement direction on screen

    ## draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)  # overlay speed and distance metrics for each object

    # Save video
    save_video(output_video_frames, 'output_videos/output_video_2.avi')

if __name__ == '__main__':
    main()  
