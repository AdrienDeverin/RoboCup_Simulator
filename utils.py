import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms
from constantes_physique import *
import math

def normalize(v):
    """Normalise un vecteur."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def angle_between_vectors(v1, v2):
    """Calcule l'angle entre deux vecteurs en radians."""
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product)

def rotate_vector(v, angle):
    """Rotation 2D d'un vecteur par un angle donné."""
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(rotation_matrix, v)


def minimal_arc_length_on_circle(R: float, 
                                 x1: float, y1: float, 
                                 x2: float, y2: float) -> float:
    """
    Returns the minimal distance along the circle of radius R
    (CENTERED AT THE ORIGINE) between points (x1,y1) and (x2,y2).
    Both points are assumed to lie on the circle: sqrt(x^2 + y^2) = R.
    """
    # 1) Compute the angles (in radians) of each point
    theta1 = math.atan2(y1, x1)
    theta2 = math.atan2(y2, x2)
    
    # 2) Find the absolute difference in angles
    dtheta = abs(theta2 - theta1)
    
    # 3) The minimal angular distance can't exceed pi (for the "short" arc)
    #    so we take the smaller arc between dtheta and 2π - dtheta.
    dtheta = min(dtheta, 2*math.pi - dtheta)
    
    # 4) Arc length = radius * angle (in radians)
    arc_length = R * dtheta
    return arc_length

def distance_on_circle(point1, point2, radius):
    """
    Calculate the distance between two points on a circle.

    :param point1: Tuple (x1, y1) representing the first point.
    :param point2: Tuple (x2, y2) representing the second point.
    :param radius: Radius of the circle.
    :return: The distance between the two points on the circle.
    """
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the Euclidean distance between the two points
    chord_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Calculate the central angle theta using the chord length formula
    # chord_length = 2 * r * sin(theta / 2)
    # Solving for theta: theta = 2 * arcsin(chord_length / (2 * r))
    theta = 2 * math.asin(chord_length / (2 * radius))

    # Calculate the arc length (distance on the circle)
    distance = radius * theta
    return distance

def avancer_sur_cercle(C, R, point_initial, w, d):
    """
    Avance un point sur un cercle d'une distance d.
    Retourne :
    - tuple, les coordonnées du nouveau point (Px', Py')
    """
    # Extraire les coordonnées du centre et du point initial
    Cx, Cy = C
    Px, Py = point_initial

    # Calculer l'angle initial du point par rapport au centre
    angle_initial = math.atan2(Py - Cy, Px - Cx)

    # Calculer l'angle parcouru en fonction de la distance d
    angle_parcouru = w * (d / R)

    # Calculer le nouvel angle
    nouvel_angle = angle_initial + angle_parcouru

    # Calculer les nouvelles coordonnées
    Px_prime = Cx + R * math.cos(nouvel_angle)
    Py_prime = Cy + R * math.sin(nouvel_angle)

    return (Px_prime, Py_prime)

def fast_normalized_vector(p1, p2):
    v = p2 - p1
    norm_sq = v @ v  # Faster dot product
    if norm_sq < 1e-8:  # avoid division by zero
        return np.zeros_like(v)
    return v/ np.sqrt(norm_sq)  
    
def Affichage_Historique(trajectory_ball, speeds_ball, 
                         ennemis_trajectories, ennemis_direction, ennemis_velocity_records, 
                         allies_trajectories, allies_direction, allies_velocity_records,
                         time_step_affichage_ms, time_step_ms):
    trajectory_ball = np.array(trajectory_ball)
    plt.figure(figsize=(10, 6))
    # FIELD
    plt.gca().add_patch(Rectangle((0, 0), FIELD_DIMENSIONS[0], FIELD_DIMENSIONS[1], fill=True, color="green", linewidth=2)) # Gason
    plt.gca().add_patch(Rectangle((0, (FIELD_DIMENSIONS[1] - GOAL_SIZE) / 2), GOAL_DEPTH, GOAL_SIZE, fill=False, edgecolor="white", linewidth=2))  # Goal Allie - Left side
    plt.gca().add_patch(Rectangle((FIELD_DIMENSIONS[0]- GOAL_DEPTH, (FIELD_DIMENSIONS[1] - GOAL_SIZE) / 2), GOAL_DEPTH, GOAL_SIZE, fill=False, edgecolor="white", linewidth=2)) # Goal Ennemie - Right side

    ## BALL ##
    plt.plot(trajectory_ball[:, 0], trajectory_ball[:, 1], label="Trajectoire Balle", color="blue") # Dessin de la trajectoire balle 
    plt.gca().add_patch(Circle(trajectory_ball[-1], BALL_RADIUS, color="#FF69B4", fill=True, zorder=2)) # Ajoute position balle finale
    for i, pos in enumerate(trajectory_ball):
        plt.gca().add_patch(Circle(pos, BALL_RADIUS/2, color="#FF69B4", fill=True, zorder=2)) # Ajoute position balle intermédiaire
        plt.text(pos[0] + 0.3, pos[1] - 0.3, f"{speeds_ball[i]:.2f} m/s", fontsize=8, color="black", zorder=4) # Ajout de la vitesse en texte 

    ## ENNEMIS ##
    for i, traj in ennemis_trajectories.items():
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], linestyle="--", color="red") # Dessin de la trajectoire du robot
        rect = Rectangle((traj[-1, 0] - ROBOTS_RADIUS, traj[-1, 1] - ROBOTS_RADIUS), 2*ROBOTS_RADIUS, 2*ROBOTS_RADIUS, color="red", fill=True, zorder=2) # Dessin position final  
        t = transforms.Affine2D().rotate_deg_around(traj[-1, 0], traj[-1, 1], np.degrees(np.arctan2(ennemis_direction[i][-1][1], ennemis_direction[i][-1][0]))) + plt.gca().transData
        rect.set_transform(t)
        plt.gca().add_patch(rect)

    for i, vel in ennemis_velocity_records.items():
        pos = ennemis_trajectories[i][::time_step_affichage_ms // time_step_ms] 
        pos.append(ennemis_trajectories[i][-1])
        delta_x = 0.3
        for j in range(len(vel)):
            plt.quiver(pos[j][0], pos[j][1], vel[j][0] / 2, vel[j][1] / 2, angles='xy', scale_units='xy', scale=2, color="purple", width=0.003, zorder= 3) # Ajout des vecteurs vitesse en flèche
            plt.text(pos[j][0] + delta_x, pos[j][1] - 0.1, f"{np.linalg.norm(vel[j]):.2f} m/s", fontsize=8, color="orange", zorder=4) # Ajoute vitesse en texte
            x, y = pos[j][0], pos[j][1] # dessine la position intermédiaire
            rect = Rectangle((x - ROBOTS_RADIUS / 2 , y - ROBOTS_RADIUS / 2 ), ROBOTS_RADIUS, ROBOTS_RADIUS , color="red", fill=True, zorder=2)
            t = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(np.arctan2(ennemis_direction[i][j][1], ennemis_direction[i][j][0]))) + plt.gca().transData
            rect.set_transform(t)
            plt.gca().add_patch(rect)
            if(delta_x > 0): delta_x = -0.9
            else : delta_x = 0.3

     ## ALLIES ##
    for i, traj in allies_trajectories.items():
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], linestyle="--", color="red") # Dessin de la trajectoire du robot
        rect = Rectangle((traj[-1, 0] - ROBOTS_RADIUS, traj[-1, 1] - ROBOTS_RADIUS), 2*ROBOTS_RADIUS, 2*ROBOTS_RADIUS, color="red", fill=True, zorder=2) # Dessin position final  
        t = transforms.Affine2D().rotate_deg_around(traj[-1, 0], traj[-1, 1], np.degrees(np.arctan2(allies_direction[i][-1][1], allies_direction[i][-1][0]))) + plt.gca().transData
        rect.set_transform(t)
        plt.gca().add_patch(rect)

    for i, vel in allies_velocity_records.items():
        pos = allies_trajectories[i][::time_step_affichage_ms // time_step_ms] 
        pos.append(allies_trajectories[i][-1])
        for j in range(len(vel)):
            plt.quiver(pos[j][0], pos[j][1], vel[j][0] / 2, vel[j][1] / 2, angles='xy', scale_units='xy', scale=2, color="purple", width=0.003, zorder= 3) # Ajout des vecteurs vitesse en flèche
            plt.text(pos[j][0] + 0.2, pos[j][1] + 0.1, f"{np.linalg.norm(vel[j]):.2f} m/s", fontsize=8, color="orange", zorder=4) # Ajoute vitesse en texte
            x, y = pos[j][0], pos[j][1] # dessine la position intermédiaire
            rect = Rectangle((x - ROBOTS_RADIUS / 2 , y - ROBOTS_RADIUS / 2 ), ROBOTS_RADIUS, ROBOTS_RADIUS , color="red", fill=True, zorder=2)
            t = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(np.arctan2(allies_direction[i][j][1], allies_direction[i][j][0]))) + plt.gca().transData
            rect.set_transform(t)
            plt.gca().add_patch(rect)

    # Configuration du graphique
    plt.xlim(-1, FIELD_DIMENSIONS[0] + 1)
    plt.ylim(-1, FIELD_DIMENSIONS[1] + 1)
    plt.xlabel("position_balle X (m)")
    plt.ylabel("position_balle Y (m)")
    plt.title("Simulation de la trajectoire d'une balle avec obstacles")
    plt.legend()
    plt.grid()
    plt.show()
