import numpy as np

FIELD_DIMENSIONS = (22, 14)  # Largeur et hauteur du terrain en mètres
GOAL_SIZE = 6
GOAL_DEPTH = 1
BALL_RADIUS = 0.1  # Rayon de la balle en mètres
ROBOTS_RADIUS = 0.2 # Rayon robot
ROBOTS_HEIGHT = 0.8 # Taille robot
FRICTION_COEFF = 0.5  # Coefficient de frottement (décélération par seconde)
ROBOTS_MAX_SPEED = 4  # Vitesse maximale des obstacles en m/s
ROBOTS_MAX_ANGULAR_SPEED = np.pi # Vitesse angulaire maximale des obstacles en radians/s
ACCELERATION_RATE = 4  # Accélération des obstacles en m/s^2
DECELERATION_RATE = 4  # Décélération des obstacles en m/s^2
COEFFICIENT_RESTITUTION = 0.8  # Coefficient de restitution pour le rebond
GRIPPER_ABSORPTION = 1.5 # m.s-1