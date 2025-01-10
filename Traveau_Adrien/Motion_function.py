from constantes_physique import *
from utils import *
import math


def FoundGoodCircle(pos_robot, pos_target, velocity_robot, velocity_target):

    # TODO : Vitesse angulaire max (ici fixée, pourrait dépendre de la vitesse courante)
    omega_robot  = ROBOTS_MAX_ANGULAR_SPEED
    omega_target = ROBOTS_MAX_ANGULAR_SPEED

    # Rayons des cercles (v / omega)
    rayon_cercle_robot  = np.linalg.norm(velocity_robot)  / omega_robot
    rayon_cercle_target = np.linalg.norm(velocity_target) / omega_target

    # Normal à la vitesse du robot (pour construire le centre du cercle)
    normal_robot = np.array([-velocity_robot[1], velocity_robot[0]])
    norm_r = np.linalg.norm(normal_robot)
    if norm_r > 1e-12:
        normal_robot /= norm_r

    # Normal à la vitesse de la target
    normal_target = np.array([-velocity_target[1], velocity_target[0]])
    norm_t = np.linalg.norm(normal_target)
    if norm_t > 1e-12:
        normal_target /= norm_t

    # Centres possibles du robot (côté + et -)
    centre_ro_plus  = pos_robot  + normal_robot  * rayon_cercle_robot
    centre_ro_minus = pos_robot  - normal_robot  * rayon_cercle_robot

    # Centres possibles de la cible (côté + et -)
    centre_tar_plus  = pos_target + normal_target  * rayon_cercle_target
    centre_tar_minus = pos_target - normal_target  * rayon_cercle_target

    # On prépare les 4 combinaisons (robot ±, target ±)
    robot_centers  = [centre_ro_plus,  centre_ro_plus,  centre_ro_minus, centre_ro_minus]
    target_centers = [centre_tar_plus, centre_tar_minus, centre_tar_plus, centre_tar_minus]

    # Sens de rotation associés à chaque combinaison
    sens_robot  = [ 1,  1, -1, -1]
    sens_target = [ 1, -1,  1, -1]

    # Calcul des distances au carré pour chaque combo
    # On évite np.linalg.norm(...) ** 2 pour gagner un peu de temps : on fait un produit scalaire direct
    dist_sqr = []
    for i in range(4):
        diff = robot_centers[i] - target_centers[i]
        dist_sqr.append(diff.dot(diff))  # distance²

    # On prend l'indice du minimum
    min_index = np.argmin(dist_sqr)

    # Sélection de la meilleure combo
    centre_cercle_robot  = robot_centers[min_index]
    centre_cercle_target = target_centers[min_index]
    final_sens_robot     = sens_robot[min_index]
    final_sens_target    = sens_target[min_index]

    return (centre_cercle_robot,  rayon_cercle_robot,  final_sens_robot,
            centre_cercle_target, rayon_cercle_target, final_sens_target)

def tangentes_ext_1(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'une des tangentes externes entre deux cercles.
    Retourne les deux éléments, (x1,y1),(x2,y2),  points de contact sur cercle1 et cercle2.
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= |r1 - r2|
    if d < abs(r1 - r2):
        return 0,0# pas de tangentes externes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos(|r1 - r2| / d)
    alpha = math.acos(abs(r1 - r2) / d)

    # On définit gamma+
    gp = beta + alpha
    delta = math.pi/2

    # Points de contact : formules (externe)
    # Cercle 1
    x1p = cx1 - r1 * math.sin(gp+ delta)
    y1p = cy1 + r1 * math.cos(gp+ delta)

    # Cercle 2
    x2p = cx2 - r2 * math.sin(gp+ delta)
    y2p = cy2 + r2 * math.cos(gp+ delta)

    tang_plus = ((x1p, y1p), (x2p, y2p))

    return tang_plus

def tangentes_ext_2(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'autre tangentes externes entre deux cercles.
    Retourne les deux éléments, (x1,y1),(x2,y2),  points de contact sur cercle1 et cercle2.
    """

    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= |r1 - r2|
    if d < abs(r1 - r2):
        return 0,0  # pas de tangentes externes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos(|r1 - r2| / d)
    alpha = math.acos(abs(r1 - r2) / d)

    # On définit gamma-
    gm = beta - alpha
    delta = math.pi/2

    # Points de contact : formules (externe)
    # Cercle 1
    x1m = cx1 - r1 * math.sin(gm+ delta)
    y1m = cy1 + r1 * math.cos(gm+ delta)

    # Cercle 2
    x2m = cx2 - r2 * math.sin(gm+ delta)
    y2m = cy2 + r2 * math.cos(gm+ delta)

    tang_minus = ((x1m, y1m), (x2m, y2m))

    return tang_minus

def tangentes_inter_1(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'une des tangentes internes (ou 'croisées') entre deux cercles.
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= (r1 + r2)
    if d < (r1 + r2):
        return 0,0  # pas de tangentes internes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos((r1 + r2) / d)
    alpha = math.acos((r1 + r2) / d)

    # On définit gamma+ 
    gp = beta + alpha
    delta = math.pi/2
    # Points de contact : formules (interne)
    # Cercle 1
    x1p = cx1 - r1 * math.sin(gp- delta)
    y1p = cy1 + r1 * math.cos(gp- delta)

    # Cercle 2 : "r2" se comporte comme s'il était négatif
    x2p = cx2 + r2 * math.sin(gp- delta)   # + au lieu de - 
    y2p = cy2 - r2 * math.cos(gp- delta)   # - au lieu de +

    tang_plus = ((x1p, y1p), (x2p, y2p))
    return tang_plus

def tangentes_inter_2(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule l'autre tangentes internes (ou 'croisées') entre deux cercles.
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= (r1 + r2)
    if d < (r1 + r2):
        return 0,0  # pas de tangentes internes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos((r1 + r2) / d)
    alpha = math.acos((r1 + r2) / d)

    # On définit gamma-
    gm = beta - alpha
    delta = math.pi/2
    # Points de contact : formules (interne)
    # Cercle 1
    x1m = cx1 - r1 * math.sin(gm- delta)
    y1m = cy1 + r1 * math.cos(gm- delta)

    # Cercle 2 : "r2" se comporte comme s'il était négatif
    x2m = cx2 + r2 * math.sin(gm- delta)
    y2m = cy2 - r2 * math.cos(gm- delta)

    tang_minus = ((x1m, y1m), (x2m, y2m))

    return  tang_minus

def tangentes_externes(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule les 2 tangentes externes entre deux cercles.
    Retourne une liste de 2 éléments, chacun étant ((x1,y1),(x2,y2)),
    les points de contact sur cercle1 et cercle2.
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= |r1 - r2|
    if d < abs(r1 - r2):
        return []  # pas de tangentes externes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos(|r1 - r2| / d)
    alpha = math.acos(abs(r1 - r2) / d)

    # On définit gamma+ et gamma-
    gp = beta + alpha
    gm = beta - alpha
    delta = math.pi/2

    # Points de contact : formules (externe)
    # Cercle 1
    x1p = cx1 - r1 * math.sin(gp+ delta)
    y1p = cy1 + r1 * math.cos(gp+ delta)
    x1m = cx1 - r1 * math.sin(gm+ delta)
    y1m = cy1 + r1 * math.cos(gm+ delta)

    # Cercle 2
    x2p = cx2 - r2 * math.sin(gp+ delta)
    y2p = cy2 + r2 * math.cos(gp+ delta)
    x2m = cx2 - r2 * math.sin(gm+ delta)
    y2m = cy2 + r2 * math.cos(gm+ delta)

    tang_plus = ((x1p, y1p), (x2p, y2p))
    tang_minus = ((x1m, y1m), (x2m, y2m))

    return [tang_plus, tang_minus]

def tangentes_internes(cx1, cy1, r1, cx2, cy2, r2):
    """
    Calcule les 2 tangentes internes (ou 'croisées') entre deux cercles.
    Retourne une liste de 2 éléments, chacun étant ((x1,y1),(x2,y2)).
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2)

    # Vérification d'existence : d >= (r1 + r2)
    if d < (r1 + r2):
        return []  # pas de tangentes internes réelles

    # Angle de la ligne des centres
    beta = math.atan2(dy, dx)
    # alpha = arccos((r1 + r2) / d)
    alpha = math.acos((r1 + r2) / d)

    # On définit gamma+ et gamma-
    gp = beta + alpha
    gm = beta - alpha
    delta = math.pi/2
    # Points de contact : formules (interne)
    # Cercle 1
    x1p = cx1 - r1 * math.sin(gp- delta)
    y1p = cy1 + r1 * math.cos(gp- delta)
    x1m = cx1 - r1 * math.sin(gm- delta)
    y1m = cy1 + r1 * math.cos(gm- delta)

    # Cercle 2 : "r2" se comporte comme s'il était négatif
    x2p = cx2 + r2 * math.sin(gp- delta)   # + au lieu de - 
    y2p = cy2 - r2 * math.cos(gp- delta)   # - au lieu de +
    x2m = cx2 + r2 * math.sin(gm- delta)
    y2m = cy2 - r2 * math.cos(gm- delta)

    tang_plus = ((x1p, y1p), (x2p, y2p))
    tang_minus = ((x1m, y1m), (x2m, y2m))

    return [tang_plus, tang_minus]

def tangentes_2_cercles(cx1, cy1, r1, cx2, cy2, r2):
    """
    Renvoie la liste des 4 tangentes (2 externes + 2 internes)
    sous forme de 4 couples [ ((x1,y1),(x2,y2)), ... ].
    Certaines peuvent être vides si le cas géométrique ne le permet pas.
    """
    tex = tangentes_externes(cx1, cy1, r1, cx2, cy2, r2)
    tin = tangentes_internes(cx1, cy1, r1, cx2, cy2, r2)
    return tex + tin




# TODO Changement endroit colision
def handle_collision_2D(ball_velocity, ROBOTS_velocity, ball_position, ROBOTS_position):
    """
    Gère la collision entre la balle et l'ROBOTS en appliquant le coefficient de restitution.
    On considère un choc "balle" (de masse négligeable ou similaire) contre un ROBOTS
    potentiellement en mouvement (ROBOTS_velocity).
    
    Hypothèse : l'ROBOTS est bien "face" à la balle, ou on définit la normale n
    via (ball_position - ROBOTS_position).
    """

    # Vecteur normal (de l'ROBOTS vers la balle)
    diff_pos = ball_position - ROBOTS_position
    dist = np.linalg.norm(diff_pos)
    if dist < 1e-12:
        # Évite division par zéro si positions identiques
        return ball_velocity

    n_x, n_y = diff_pos / dist
    v_x, v_y = ball_velocity
    v_sx, v_sy = ROBOTS_velocity
    # Vitesse relative
    v_rx = v_x - v_sx
    v_ry = v_y - v_sy
    # Produit scalaire v_r . n
    dot_prod = v_rx * n_x + v_ry * n_y
    # Composante de restitution
    alpha = (1 + COEFFICIENT_RESTITUTION) * dot_prod
    # Nouvelle vitesse relative
    v_r_prime_x = v_rx - alpha * n_x
    v_r_prime_y = v_ry - alpha * n_y
    # Revenir au référentiel global
    v_prime_x = v_r_prime_x + v_sx
    v_prime_y = v_r_prime_y + v_sy
    return np.array([v_prime_x, v_prime_y])

def compute_collision_time(
    pos_b, vel_b,
    pos_o, vel_o,
    radius_sum,
    dt
):
    """
    Calcule l'instant t ∈ [0, dt] où la balle (pos_b, vel_b)
    touche l'ROBOTS (pos_o, vel_o),
    en considérant un rayon effectif = radius_sum = BALL_RADIUS + ROBOTS_RADIUS.

    Hypothèse : mouvement rectiligne (vitesse constante) pendant ce micro-pas.
    Retourne la première collision t_col dans [0, dt], ou None s'il n'y a pas de collision.

    Équation :
      || (pos_b + vel_b*t) - (pos_o + vel_o*t) || = radius_sum.
    """
    # On note D = (pos_b - pos_o), V = (vel_b - vel_o)
    D = pos_b - pos_o
    V = vel_b - vel_o
    R = radius_sum

    a = np.dot(V, V)         # ||V||^2
    b = 2 * np.dot(D, V)
    c = np.dot(D, D) - R*R

    # Cas particulier : si a ~ 0 => pas de mouvement relatif
    if a < 1e-12:
        return None

    # Discriminant
    disc = b*b - 4*a*c
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    candidates = []
    for t_ in (t1, t2):
        if 0 <= t_ <= dt:
            candidates.append(t_)

    if not candidates:
        return None
    return min(candidates)

def update_ROBOTS_lvl1(position_balle, ROBOTS_list, ROBOTS_velocities, delta_t):
    """
    Met à jour les positions et vitesses des ROBOTSs en fonction de la position de la balle.
    Les ROBOTSs ajustent leur angle pour se rapprocher de la balle et accélèrent/décélèrent selon la distance.
    """
    new_positions = []
    new_velocities = []

    for pos, vel in zip(ROBOTS_list, ROBOTS_velocities):
        # Direction vers la balle
        direction_to_ball = position_balle - np.array(pos)
        direction_to_ball /= np.linalg.norm(direction_to_ball) # Normalisation si distance > 0

        if np.linalg.norm(vel) == 0: # Si l'ROBOTS est complètement immobile, on lui assigne une direction initiale pour éviter la division par zéro.
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
        angle_change = np.clip(angle_to_target, -ROBOTS_MAX_ANGULAR_SPEED * delta_t,
                                ROBOTS_MAX_ANGULAR_SPEED * delta_t)
        
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
        
        # On s'assure que la vitesse reste dans [0, ROBOTS_MAX_SPEED]
        speed = np.clip(speed, 0, ROBOTS_MAX_SPEED)
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

def update_ROBOTS_lvl2(position_balle, vitesse_balle, ROBOTS_list, ROBOTS_velocities, delta_t):
    """
    Met à jour positions et vitesses des ROBOTSs pour une interception rapide de la balle.

    Approche : 
      - Prédiction d'un point futur de la balle (p_balle_future).
      - Rotation et accélération de l'ROBOTS en direction de ce point pour intercepter rapidement.
    """
    new_positions = []
    new_velocities = []

    # Petit "horizon de prédiction" (en secondes)
    TIME_PREDICT = 0.05  
    for pos, vel in zip(ROBOTS_list, ROBOTS_velocities):
        # ----------------------------------------------
        # 1) PREDICTION DE LA POSITION FUTURE DE LA BALLE
        # ----------------------------------------------
        p_balle_future = position_balle + vitesse_balle * TIME_PREDICT

        # ----------------------------------------------
        # 2) DIRECTION (ROBOTS -> p_balle_future)
        # ----------------------------------------------
        direction_to_future = p_balle_future - np.array(pos)
        dist_to_future = np.linalg.norm(direction_to_future)

        # Eviter division par zéro
        if dist_to_future > 1e-8:
            direction_to_future /= dist_to_future
        else:
            direction_to_future = np.zeros_like(direction_to_future)

        # Si l'ROBOTS est immobile, on lui assigne une direction initiale
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
        max_angle = ROBOTS_MAX_ANGULAR_SPEED * delta_t
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
        new_speed = np.clip(new_speed, 0, ROBOTS_MAX_SPEED)

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

def update_ROBOTS_lvl3(position_balle,vitesse_balle,ROBOTS_positions,ROBOTS_velocities,delta_t):
    """
    Interception rapide + prise en compte de l'angle de contrôle (rotation) 
    pour un swerve drive.

    - On prédit la position future de la balle (p_balle_future).
    - On calcule l'angle nécessaire pour s'orienter vers cette cible.
    - On estime le temps de rotation t_rot + temps de déplacement linéaire.
    - On ajuste accélération/décélération pour synchroniser l'arrivée.
    """

    new_positions = []
    new_velocities = []

    # Horizon de prédiction (vous pouvez le faire varier)
    TIME_PREDICT = 0.05

    for pos, vel in zip(ROBOTS_positions, ROBOTS_velocities):
        # ==============================
        # 1) POSITION FUTURE DE LA BALLE
        # ==============================
        p_balle_future = position_balle + vitesse_balle * TIME_PREDICT

        # ==============================
        # 2) VECTEUR / DISTANCE / DIRECTION VERS CETTE CIBLE
        # ==============================
        vec_to_future = p_balle_future - np.array(pos)
        dist_to_future = np.linalg.norm(vec_to_future)

        if dist_to_future > 1e-8:
            dir_to_future = vec_to_future / dist_to_future
        else:
            dir_to_future = np.zeros_like(vec_to_future)

        # Initialisation si vitesse nulle
        if np.linalg.norm(vel) < 1e-8:
            # "boost" initial
            vel = dir_to_future * ACCELERATION_RATE * delta_t

        # ==============================
        # 3) DETERMINATION DE L'ANGLE COURANT + ANGLE CIBLE
        # ==============================
        #   - Angle courant (theta_cur) : on le déduit de la vitesse vel
        #     si le châssis du swerve drive s'oriente dans le sens de vel.
        #   - Angle cible (theta_tar)   : direction de dir_to_future
        # ==============================
        current_speed = np.linalg.norm(vel)
        if current_speed > 1e-8:
            current_dir = vel / current_speed
        else:
            current_dir = np.zeros_like(vel)

        # angle courant
        theta_cur = np.arctan2(current_dir[1], current_dir[0])

        # angle cible
        theta_tar = np.arctan2(dir_to_future[1], dir_to_future[0])

        # Différence d'angle dans [-pi, pi]
        angle_diff = theta_tar - theta_cur
        # normalisation
        if angle_diff > np.pi:
            angle_diff -= 2*np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2*np.pi

        # ==============================
        # 4) ROTATION EFFECTIVE DU SWERVE
        # ==============================
        #   - On limite la rotation max sur ce delta_t.
        # ==============================
        max_angle = ROBOTS_MAX_ANGULAR_SPEED * delta_t
        angle_change = np.clip(angle_diff, -max_angle, max_angle)

        # Calcul de la nouvelle direction via matrice de rotation
        cos_a = np.cos(angle_change)
        sin_a = np.sin(angle_change)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        new_velocity = rotation_matrix.dot(vel)

        # ==============================
        # 5) CALCUL DU TEMPS D'INTERCEPTION ROBOTS
        # ==============================
        #   => t_rot + t_lin
        # ==============================
        # (a) temps de rotation "restant" (approx) : 
        #     on suppose que l'ROBOTS a encore angle_diff à rattraper,
        #     mais angle_change en a déjà été fait. 
        #     On peut l'approximer par:
        remaining_angle = angle_diff - angle_change
        # normaliser de nouveau
        if remaining_angle > np.pi:
            remaining_angle -= 2*np.pi
        elif remaining_angle < -np.pi:
            remaining_angle += 2*np.pi

        t_rot = abs(remaining_angle) / ROBOTS_MAX_ANGULAR_SPEED

        # (b) temps de déplacement linéaire 
        #     on suppose qu'il va se déplacer à la vitesse `obs_speed` (ou max) vers p_balle_future
        obs_speed = np.linalg.norm(new_velocity)
        if obs_speed < 1e-8:
            t_lin = 9999.0
        else:
            t_lin = dist_to_future / obs_speed

        t_obs = t_rot + t_lin

        # ==============================
        # 6) CALCUL DU TEMPS D'ARRIVEE DE LA BALLE
        # ==============================
        ball_speed = np.linalg.norm(vitesse_balle)
        if ball_speed < 1e-8:
            # balle quasi immobile => "elle n'arrivera pas" 
            t_balle = 9999.0
        else:
            t_balle = dist_to_future / ball_speed

        # ==============================
        # 7) DECISION D'ACCELERER OU DE FREINER
        # ==============================
        speed_val = obs_speed
        # On autorise un epsilon pour éviter des oscillations en permanence
        EPSILON_SYNC = 0.05

        if t_obs > t_balle + EPSILON_SYNC:
            # L'ROBOTS arrivera trop tard => accélère
            speed_val += ACCELERATION_RATE * delta_t
        elif t_obs < t_balle - EPSILON_SYNC:
            # L'ROBOTS arrivera trop tôt => ralentit
            speed_val -= DECELERATION_RATE * delta_t
        else:
            # On est dans la bonne fenêtre => stabiliser
            pass

        # Limitation de la vitesse
        speed_val = np.clip(speed_val, 0, ROBOTS_MAX_SPEED)

        if speed_val > 1e-8 and np.linalg.norm(new_velocity) > 1e-8:
            new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * speed_val
        else:
            new_velocity = np.zeros_like(new_velocity)

        # ==============================
        # 8) MISE A JOUR DE LA POSITION
        # ==============================
        new_position = np.array(pos) + new_velocity * delta_t

        # Stockage
        new_positions.append(tuple(new_position))
        new_velocities.append(new_velocity)

    return new_positions, new_velocities
