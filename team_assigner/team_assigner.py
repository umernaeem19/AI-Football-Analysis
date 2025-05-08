from sklearn.cluster import KMeans  # importing KMeans for clustering colors (used to separate teams)
import numpy as np
class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # stores average color of each team after clustering
        self.player_team_dict = {}  # stores which team each player belongs to, keyed by player ID
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)  # convert the image into a flat list of RGB pixels
        
        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)  # try to group into 2 dominant colors
        kmeans.fit(image_2d)  

        return kmeans  

    def get_player_color(self, frame, bbox):
        # crop the image using the bounding box of the player
        x1, y1, x2, y2 = bbox

        # Apply np.floor for the top-left corner and np.ceil for the bottom-right
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))

        # Ensure bbox is within image bounds
        h, w, _ = frame.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # If the bbox has no area, skip it
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Skipping bbox with zero area: {bbox}")
            return np.array([0, 0, 0])

        # Crop the image
        image = frame[y1:y2, x1:x2]

        # Take only the top half of the image to focus on the jersey (avoid shorts and legs)
        top_half_image = image[0:int(image.shape[0]/2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)  # cluster the cropped region

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        # Check background by using corners as they are likely part of the background
        corner_clusters = [ 
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)  # find most common background cluster
        player_cluster = 1 - non_player_cluster  # assume the other cluster belongs to the player

        player_color = kmeans.cluster_centers_[player_cluster]  # get average color of the player's cluster

        return player_color 

    def assign_team_color(self, frame, player_detections):
        # extract player colors from the first frame
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]  # get bounding box
            player_color = self.get_player_color(frame, bbox)  # extract dominant jersey color
            player_colors.append(player_color)

        # cluster all player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans  # save model to use later

        # store each team’s representative color
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # if we've already assigned this player to a team, return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # otherwise, compute the player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # use the trained k-means model to predict which team color this player belongs to
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # shift from 0-indexed to 1-indexed

        # special handling (manual override) for player 91 to force them into team 1
        if player_id == 91:
            team_id = 1

        # store this assignment so we don’t compute it again later
        self.player_team_dict[player_id] = team_id

        return team_id  # return the assigned team number
