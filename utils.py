import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms
from constantes_physique import *

def generate_random_velocity(max_speed):
    speed = np.random.uniform(0, max_speed)
    angle = np.random.uniform(0, 2 * np.pi)
    vel_x = speed * np.cos(angle)
    vel_y = speed * np.sin(angle)
    return np.array([vel_x, vel_y])

def acceleration(vel):
    speed = np.linalg.norm(vel)
    return -FRICTION_COEFF * vel / speed if speed > 0 else np.zeros_like(vel)

# approximation rk4
def update_ball_speed(pos, vel, dt):
    k1_v = acceleration(vel) * dt
    k1_p = vel * dt
    k2_v = acceleration(vel + k1_v / 2) * dt
    k2_p = (vel + k1_v / 2) * dt
    k3_v = acceleration(vel + k2_v / 2) * dt
    k3_p = (vel + k2_v / 2) * dt
    k4_v = acceleration(vel + k3_v) * dt
    k4_p = (vel + k3_v) * dt

    new_vel = vel + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    new_pos = pos + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
    return new_pos, new_vel

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
