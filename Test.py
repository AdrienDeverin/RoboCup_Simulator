import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms

# -----------------------------
# CONSTANTES ET PARAMÈTRES
# -----------------------------
FIELD_DIMENSIONS = (15, 10)   # Largeur, hauteur (m)
BALL_RADIUS = 0.1
OBSTACLE_RADIUS = 0.2
WIDTH, HEIGHT = 0.4, 0.4     # Taille (rectangle) pour l'affichage obstacle
FRICTION_COEFF = 0.5         # Décélération proportionnelle à la vitesse (m/s^2 par m/s)
OBSTACLE_MAX_SPEED = 4.0
OBSTACLE_MAX_ANGULAR_SPEED = np.pi
ACCELERATION_RATE = 4.0
DECELERATION_RATE = 4.0
COEFFICIENT_RESTITUTION = 0.8


# -----------------------------
# 2) FONCTION D'ACCÉLÉRATION (pour la balle)
# -----------------------------
def acceleration(velocity):
    """
    Retourne l'accélération de la balle (ex: frottement).
    Vecteur proportionnel à -velocity.
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-12:
        return np.zeros_like(velocity)
    # Frottement = -FRICTION * direction
    return -FRICTION_COEFF * velocity / speed


# -----------------------------
# 3) MISE À JOUR RK4 (pour la balle)
# -----------------------------
def update_rk4(pos, vel, dt):
    """
    Met à jour (pos, vel) sur un pas dt en utilisant un schéma RK4
    pour la balle soumise à l'accélération 'acceleration(vel)'.
    """
    # k1
    k1_v = acceleration(vel) * dt
    k1_p = vel * dt

    # k2
    v2 = vel + k1_v / 2
    k2_v = acceleration(v2) * dt
    k2_p = v2 * dt

    # k3
    v3 = vel + k2_v / 2
    k3_v = acceleration(v3) * dt
    k3_p = v3 * dt

    # k4
    v4 = vel + k3_v
    k4_v = acceleration(v4) * dt
    k4_p = v4 * dt

    new_vel = vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    new_pos = pos + (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6

    return new_pos, new_vel


# -----------------------------
# 4) COLLISION BALL - OBSTACLE
# -----------------------------
def handle_collision(ball_vel, obs_vel, ball_pos, obs_pos):
    """
    Gère la collision (rebond) balle–obstacle (modèle élastique partiel).
    """
    diff_pos = ball_pos - obs_pos
    dist = np.linalg.norm(diff_pos)
    if dist < 1e-12:
        # positions quasi identiques => éviter division par zéro
        return ball_vel.copy()

    # Normale de (obs -> balle)
    nx, ny = diff_pos / dist

    # Vitesse relative
    vrx = ball_vel[0] - obs_vel[0]
    vry = ball_vel[1] - obs_vel[1]

    # Produit scalaire v_r . n
    dot = vrx * nx + vry * ny

    # Facteur restitution
    alpha = (1 + COEFFICIENT_RESTITUTION) * dot

    # Nouvelle vitesse relative
    vrx_prime = vrx - alpha * nx
    vry_prime = vry - alpha * ny

    # Retour référentiel global
    vx_prime = vrx_prime + obs_vel[0]
    vy_prime = vry_prime + obs_vel[1]

    return np.array([vx_prime, vy_prime], dtype=float)


# -----------------------------
# 5) TEMPS DE COLLISION
# -----------------------------
def compute_collision_time(
    pos_b, vel_b,
    pos_o, vel_o,
    radius_sum, dt
):
    """
    Cherche t ∈ [0, dt] pour ||(pos_b + vel_b*t) - (pos_o + vel_o*t)|| = radius_sum.
    Renvoie None si pas de collision dans [0, dt].
    Mouvement supposé rectiligne (vel constant) pour ce sous-pas.
    """
    D = pos_b - pos_o
    V = vel_b - vel_o
    R = radius_sum

    a = np.dot(V, V)
    b = 2*np.dot(D, V)
    c = np.dot(D, D) - R*R

    if a < 1e-14:
        # Pas de mouvement relatif
        return None

    disc = b*b - 4*a*c
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc)/(2*a)
    t2 = (-b + sqrt_disc)/(2*a)

    t_candidates = []
    for t_ in (t1, t2):
        if 0 <= t_ <= dt:
            t_candidates.append(t_)

    if not t_candidates:
        return None
    return min(t_candidates)


# -----------------------------
# 6) MISE À JOUR OBSTACLE (lvl3, par ex.)
# -----------------------------

def update_obstacle_lvl1(position_balle, obstacle_list, obstacle_velocities, delta_t):
    """
    Met à jour les positions et vitesses des obstacles en fonction de la position de la balle.
    Les obstacles ajustent leur angle pour se rapprocher de la balle et accélèrent/décélèrent selon la distance.
    """
    new_positions = []
    new_velocities = []

    for pos, vel in zip(obstacle_list, obstacle_velocities):
        # Direction vers la balle
        direction_to_ball = position_balle - np.array(pos)
        direction_to_ball /= np.linalg.norm(direction_to_ball) # Normalisation si distance > 0

        if np.linalg.norm(vel) == 0: # Si l'obstacle est complètement immobile, on lui assigne une direction initiale pour éviter la division par zéro.
            vel = direction_to_ball * (ACCELERATION_RATE * delta_t)  
          
        # Direction actuelle normalisée (pour le calcul d'angle)
        current_speed = np.linalg.norm(vel)
        current_direction = vel / current_speed
       
        # Angle entre la direction actuelle et la direction vers la balle
        dot_product = np.dot(current_direction, direction_to_ball)
        cross_product = np.cross(np.append(current_direction, 0),
                                np.append(direction_to_ball, 0))[-1]
        angle_to_target = np.arctan2(cross_product, dot_product)
        
        # Limiter la rotation selon la vitesse angulaire max
        angle_change = np.clip(angle_to_target, -OBSTACLE_MAX_ANGULAR_SPEED * delta_t,
                                OBSTACLE_MAX_ANGULAR_SPEED * delta_t)
        
        # Mise à jour de la direction de la vitesse
        rotation_matrix = np.array([[np.cos(angle_change), -np.sin(angle_change)],
                                    [np.sin(angle_change),  np.cos(angle_change)]])
        new_velocity = np.dot(rotation_matrix, vel)
        
        # Accélération ou décélération selon la distance
        speed = np.linalg.norm(new_velocity)
        if np.linalg.norm(position_balle - np.array(pos)) > 2:
            speed += ACCELERATION_RATE * delta_t
        else:
            speed -= DECELERATION_RATE * delta_t
        
        # On s'assure que la vitesse reste dans [0, OBSTACLE_MAX_SPEED]
        speed = np.clip(speed, 0, OBSTACLE_MAX_SPEED)
        if speed > 0 and np.linalg.norm(new_velocity) > 0:
            new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * speed
        else:
            new_velocity = np.zeros_like(new_velocity)
        
        # Mise à jour de la position
        new_position = np.array(pos) + new_velocity * delta_t
        
        # On stocke les résultats
        new_positions.append(tuple(new_position))
        new_velocities.append(new_velocity)
    
    return new_positions, new_velocities

def update_obstacle_lvl2(position_balle, vitesse_balle, obstacle_list, obstacle_velocities, delta_t):
    """
    Met à jour positions et vitesses des obstacles pour une interception rapide de la balle.

    Approche : 
      - Prédiction d'un point futur de la balle (p_balle_future).
      - Rotation et accélération de l'obstacle en direction de ce point pour intercepter rapidement.
    """
    new_positions = []
    new_velocities = []

    # Petit "horizon de prédiction" (en secondes)
    TIME_PREDICT = 0.05  
    for pos, vel in zip(obstacle_list, obstacle_velocities):
        # ----------------------------------------------
        # 1) PREDICTION DE LA POSITION FUTURE DE LA BALLE
        # ----------------------------------------------
        p_balle_future = position_balle + vitesse_balle * TIME_PREDICT

        # ----------------------------------------------
        # 2) DIRECTION (obstacle -> p_balle_future)
        # ----------------------------------------------
        direction_to_future = p_balle_future - np.array(pos)
        dist_to_future = np.linalg.norm(direction_to_future)

        # Eviter division par zéro
        if dist_to_future > 1e-8:
            direction_to_future /= dist_to_future
        else:
            direction_to_future = np.zeros_like(direction_to_future)

        # Si l'obstacle est immobile, on lui assigne une direction initiale
        if np.linalg.norm(vel) < 1e-8:
            # Petit coup de boost dans la direction de l'interception
            vel = direction_to_future * ACCELERATION_RATE * delta_t

        # ----------------------------------------------
        # 3) Calcul de l'angle entre direction courante et direction_to_future
        # ----------------------------------------------
        current_speed = np.linalg.norm(vel)
        if current_speed > 1e-8:
            current_dir = vel / current_speed
        else:
            current_dir = np.zeros_like(vel)

        dot_product = np.dot(current_dir, direction_to_future)
        cross_product = np.cross(
            np.append(current_dir, 0),
            np.append(direction_to_future, 0) )[-1]

        angle_to_target = np.arctan2(cross_product, dot_product)

        # Limiter la rotation
        max_angle = OBSTACLE_MAX_ANGULAR_SPEED * delta_t
        angle_change = np.clip(angle_to_target, -max_angle, max_angle)

        # Calcul de la nouvelle direction via matrice de rotation
        cos_a = np.cos(angle_change)
        sin_a = np.sin(angle_change)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a] ])

        new_velocity = rotation_matrix.dot(vel)

        # ----------------------------------------------
        # 4) Accélération / Décélération
        #    On accélère si on est encore loin de la position future,
        #    sinon on ralentit pour arriver plus précisément.
        # ----------------------------------------------
        new_speed = np.linalg.norm(new_velocity)

        # Seuil de distance "proche" ou "loin"
        # On peut varier ce seuil pour ajuster la façon de s'approcher
        DIST_THRESHOLD = 1.0

        if dist_to_future > DIST_THRESHOLD:
            # Loin => on accélère
            new_speed += ACCELERATION_RATE * delta_t
        else:
            # Proche => on freine
            new_speed -= DECELERATION_RATE * delta_t

        # Clamp la vitesse
        new_speed = np.clip(new_speed, 0, OBSTACLE_MAX_SPEED)

        if new_speed > 1e-8 and np.linalg.norm(new_velocity) > 1e-8:
            new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * new_speed
        else:
            new_velocity = np.zeros_like(new_velocity)

        # ----------------------------------------------
        # 5) Mise à jour de la position
        # ----------------------------------------------
        new_position = np.array(pos) + new_velocity * delta_t

        # Stockage
        new_positions.append(tuple(new_position))
        new_velocities.append(new_velocity)

    return new_positions, new_velocities

def update_obstacle_lvl3(ball_pos, ball_vel, obs_positions, obs_velocities, dt):
    """
    Mise à jour "Swerve drive" intercept lvl3 sur un *pas de temps* dt.
    On renvoie les positions, vitesses mises à jour.
    """
    new_positions = []
    new_velocities = []

    TIME_PREDICT = 0.05
    for pos, vel in zip(obs_positions, obs_velocities):
        # 1) Position future balle (approx)
        p_balle_future = ball_pos + ball_vel * TIME_PREDICT

        # 2) Direction vers p_balle_future
        to_future = p_balle_future - pos
        dist_to_future = np.linalg.norm(to_future)
        if dist_to_future > 1e-8:
            dir_to_future = to_future / dist_to_future
        else:
            dir_to_future = np.zeros(2)

        # Init si l'obstacle est immobile
        if np.linalg.norm(vel) < 1e-8:
            vel = dir_to_future * ACCELERATION_RATE * dt

        # Angle courant
        spd = np.linalg.norm(vel)
        if spd > 1e-8:
            cur_dir = vel / spd
        else:
            cur_dir = np.zeros(2)

        theta_cur = np.arctan2(cur_dir[1], cur_dir[0])
        theta_tar = np.arctan2(dir_to_future[1], dir_to_future[0])

        angle_diff = theta_tar - theta_cur
        # normaliser dans [-pi, pi]
        if angle_diff > np.pi:
            angle_diff -= 2*np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2*np.pi

        max_angle = OBSTACLE_MAX_ANGULAR_SPEED * dt
        angle_change = np.clip(angle_diff, -max_angle, max_angle)

        # Rotation
        cos_a = np.cos(angle_change)
        sin_a = np.sin(angle_change)
        rot_m = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        vel_new = rot_m.dot(vel)

        # temps approx => t_rot + t_lin
        remain_angle = angle_diff - angle_change
        if remain_angle > np.pi:
            remain_angle -= 2*np.pi
        elif remain_angle < -np.pi:
            remain_angle += 2*np.pi

        t_rot = abs(remain_angle)/OBSTACLE_MAX_ANGULAR_SPEED
        obs_spd = np.linalg.norm(vel_new)
        if obs_spd < 1e-8:
            t_lin = 9999
        else:
            t_lin = dist_to_future/obs_spd

        t_obs = t_rot + t_lin

        # temps pour la balle
        ball_spd = np.linalg.norm(ball_vel)
        if ball_spd < 1e-8:
            t_ball = 9999
        else:
            t_ball = dist_to_future / ball_spd

        # Décision accel/frein
        speed_val = obs_spd
        EPSILON = 0.05
        if t_obs > t_ball + EPSILON:
            speed_val += ACCELERATION_RATE * dt
        elif t_obs < t_ball - EPSILON:
            speed_val -= DECELERATION_RATE * dt

        speed_val = np.clip(speed_val, 0, OBSTACLE_MAX_SPEED)
        if speed_val > 1e-8:
            vel_norm = np.linalg.norm(vel_new)
            if vel_norm > 1e-8:
                vel_new = (vel_new / vel_norm) * speed_val
            else:
                vel_new = np.zeros(2)

        # Mise à jour position
        pos_new = pos + vel_new * dt

        new_positions.append(pos_new)
        new_velocities.append(vel_new)

    return new_positions, new_velocities


# -----------------------------
# 7) SIMULATION COMPLÈTE
# -----------------------------
def simulate_ball_trajectory(initial_position, angle_deg_tir, initial_speed,
                             obstacles_pos=None,
                             obstacles_initial_velocities=None,
                             time_step_ms=50,
                             time_step_affichage_ms=200):
    """
    Simule la balle + obstacles avec collisions cohérentes.
    """
    if obstacles_pos is None:
        obstacles_pos = []
    field_w, field_h = FIELD_DIMENSIONS
    dt = time_step_ms / 1000.0

    # Balle
    pos_b = np.array(initial_position, dtype=float)
    ang_rad = np.radians(angle_deg_tir)
    vel_b = np.array([initial_speed*np.cos(ang_rad),
                      initial_speed*np.sin(ang_rad)], dtype=float)
    vel_init = vel_b.copy()

    # Obstacles
    if obstacles_initial_velocities is None:
        obs_vels = [np.zeros(2) for _ in obstacles_pos]
    else:
        obs_vels = [np.array(v, dtype=float) for v in obstacles_initial_velocities]

    # Historique
    time_last_record = 0.0
    trajectory_ball = [tuple(pos_b)]
    speeds_ball = [np.linalg.norm(vel_b)]
    obstacles_trajectories = {i: [tuple(o)] for i, o in enumerate(obstacles_pos)}
    obstacles_velocity_records = {i: [tuple(obs_vels[i])] for i in range(len(obstacles_pos))}

    # Boucle de simulation
    while np.linalg.norm(vel_b) > 1e-2:
        time_remaining = dt

        # SUBSTEP -> On répète tant qu'il reste du temps dans ce pas
        while time_remaining > 1e-8:
            # 1) Trouver la première collision t_col sur [0, time_remaining] 
            #    entre la balle et chacun des obstacles
            t_col_min = None
            col_idx = None

            for i, (o_pos, o_vel) in enumerate(zip(obstacles_pos, obs_vels)):
                # Cherche collision
                t_col = compute_collision_time(
                    pos_b, vel_b,
                    o_pos, o_vel,
                    BALL_RADIUS + OBSTACLE_RADIUS,
                    time_remaining
                )
                if t_col is not None:
                    if t_col_min is None or t_col < t_col_min:
                        t_col_min = t_col
                        col_idx = i

            if t_col_min is None:
                # => Pas de collision => on avance tout le monde sur time_remaining
                # Balle
                pos_b, vel_b = update_rk4(pos_b, vel_b, time_remaining)
                # Obstacles
                # obstacles_pos, obs_vels = update_obstacle_lvl3(
                #     pos_b, vel_b,
                #     obstacles_pos, obs_vels,
                #     time_remaining
                # )
                obstacles_pos, obs_vels = update_obstacle_lvl1(
                    pos_b, 
                    obstacles_pos, obs_vels,
                    time_remaining
                )
                # Fin du sous-step
                time_remaining = 0.0
            else:
                # => On a t_col_min
                # 2) On avance balle + obstacles jusqu'à t_col_min
                pos_b, vel_b = update_rk4(pos_b, vel_b, t_col_min)
                # Mise à jour de *tous* les obstacles, 
                # pour EXACTEMENT t_col_min
                # obstacles_pos, obs_vels = update_obstacle_lvl3(
                #     pos_b, vel_b,
                #     obstacles_pos, obs_vels,
                #     t_col_min
                # )
                obstacles_pos, obs_vels = update_obstacle_lvl1(
                    pos_b, 
                    obstacles_pos, obs_vels,
                    time_remaining
                )
                # 3) On applique la collision avec l'obstacle col_idx
                #    On recalcule la position EXACTE de cet obstacle
                obs_pos_collision = obstacles_pos[col_idx]
                obs_vel_collision = obs_vels[col_idx]
                # Rebond
                vel_b = handle_collision(vel_b, obs_vel_collision, pos_b, obs_pos_collision)

                # 4) On retire t_col_min
                time_remaining -= t_col_min

        # Vérif collisions limites (terrain)
        if ((pos_b[0] - BALL_RADIUS) < 0 or
            (pos_b[0] + BALL_RADIUS) > field_w or
            (pos_b[1] - BALL_RADIUS) < 0 or
            (pos_b[1] + BALL_RADIUS) > field_h):
            # On stoppe la balle (ex: collision mur)
            vel_b[:] = 0.0

        # Enregistrement
        trajectory_ball.append(tuple(pos_b))
        speeds_ball.append(np.linalg.norm(vel_b))
        for i, (pobs, vobs) in enumerate(zip(obstacles_pos, obs_vels)):
            obstacles_trajectories[i].append(tuple(pobs))
            # On n'enregistre la vitesse que toutes "time_step_affichage_ms" si on veut 
            # simplifier, ou à chaque pas
            obstacles_velocity_records[i].append(tuple(vobs))

        # Condition d'arrêt
        if np.linalg.norm(vel_b) < 1e-2:
            break

    # ----------- FIN BOUCLE -----------
    trajectory_ball = np.array(trajectory_ball)

    # AFFICHAGE
    plt.figure(figsize=(10, 6))
    # Terrain
    plt.gca().add_patch(Rectangle((0, 0), field_w, field_h,
                                  fill=True, color="green", zorder=1))

    # Trajectoire balle
    plt.plot(trajectory_ball[:, 0], trajectory_ball[:, 1],
             color="blue", label="Trajectoire balle")
    # Balle finale
    circ = Circle(trajectory_ball[-1], BALL_RADIUS,
                  color="orange", fill=True, zorder=5)
    plt.gca().add_patch(circ)

    # Vecteur initial
    plt.quiver(trajectory_ball[0, 0], trajectory_ball[0, 1],
               vel_init[0], vel_init[1],
               angles='xy', scale_units='xy', scale=4,
               color="black", width=0.003, zorder=6)

    # Ajout de petits cercles + vitesses
    # (on espace l'affichage)
    step_visu = max(1, time_step_affichage_ms // time_step_ms)
    for i in range(0, len(trajectory_ball), step_visu):
        pos_ = trajectory_ball[i]
        spd_ = speeds_ball[i] if i < len(speeds_ball) else 0.0
        plt.gca().add_patch(Circle(pos_, BALL_RADIUS/2,
                                   color="orange", fill=True, zorder=4))
        plt.text(pos_[0]+0.1, pos_[1], f"{spd_:.2f} m/s",
                 fontsize=8, color="black", zorder=7)

    # Obstacles
    for i, traj_list in obstacles_trajectories.items():
        arr_traj = np.array(traj_list)
        plt.plot(arr_traj[:, 0], arr_traj[:, 1],
                 '--', color="red", label=f"Obstacle {i}")
        # Obstacle final
        xobs, yobs = arr_traj[-1]
        rect = Rectangle((xobs - WIDTH/2, yobs - HEIGHT/2),
                         WIDTH, HEIGHT, color="red", fill=True, zorder=3)
        # Orientation : dernier vecteur
        vfinal = obstacles_velocity_records[i][-1]
        angle_deg = np.degrees(np.arctan2(vfinal[1], vfinal[0])) \
            if np.linalg.norm(vfinal) > 1e-8 else 0.0
        t_tr = transforms.Affine2D().rotate_deg_around(xobs, yobs, angle_deg)
        rect.set_transform(t_tr + plt.gca().transData)
        plt.gca().add_patch(rect)

        # Vecteurs vitesses (optionnel)
        vel_list = obstacles_velocity_records[i]
        for j in range(0, len(vel_list), step_visu):
            vx_, vy_ = vel_list[j]
            px_, py_ = arr_traj[j]
            plt.quiver(px_, py_, vx_, vy_, angles='xy', scale_units='xy',
                       scale=4, color="purple", width=0.003, zorder=6)

    plt.xlim(-1, field_w + 1)
    plt.ylim(-1, field_h + 1)
    plt.title("Simulation Balle + Obstacles avec collisions cohérentes (substeps)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.legend()
    plt.show()


# -----------------------------
# 8) EXEMPLE D'UTILISATION
# -----------------------------
if __name__ == "__main__":
    # Exemple : Balle tirée depuis (1,1) à 45° et 6 m/s
    # Un obstacle à (8,6) avec vitesse initiale [-2,0]
    simulate_ball_trajectory(
        initial_position=(1, 1),
        angle_deg_tir=45,
        initial_speed=6.0,
        obstacles_pos=[(8, 6)],
        obstacles_initial_velocities=[[-2.0, 0.0]],
        time_step_ms=30,
        time_step_affichage_ms=300
    )
