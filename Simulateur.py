import numpy as np
from utils import *
from objects import BALL, ROBOT
from constantes_physique import *
from Motion_function import *

initial_speed=6 # m.s-1
angle_rad = np.radians(45) # 45°
ball = BALL(pos = np.array([1.0, 1.0]), 
            velocity= np.array([initial_speed * np.cos(angle_rad), initial_speed * np.sin(angle_rad)], dtype=float), 
            iscatch=False)

ball2 = BALL(pos = np.array([15.0, 12.0]), 
            velocity= np.array([0,0], dtype=float), 
            iscatch=False)

robot_A1 = ROBOT(pos = np.array([1.0, 1.0]), 
                velocity = np.array([0 , 0]), 
                normal = np.array([1.0, 1.0]), 
                isON = True, hasBall = True)

robot_E1 = ROBOT(pos = np.array([8.0, 6.0]), 
                velocity = np.array([0 , 2]), 
                normal = np.array([-2.0, 0.0]), 
                isON = True, hasBall = True)


robot_E2 = ROBOT(pos = np.array([8.0, 6.0]), 
                velocity = np.array([0 , 2]), 
                normal = np.array([-2.0, 0.0]), 
                isON = True, hasBall = True)

def simulate_ball_trajectory(ball, robots_ennemi, robots_allie, time_step_ms=50, time_step_affichage_ms=200):
    
    # Dimensions limite affichage
    field_width, field_height = FIELD_DIMENSIONS
    dt = time_step_ms / 1000  # Conversion ms -> s

    for robot in robots_allie :
        if robot.hasBall == True :
            ball.pos = robot.pos + robot.normal * (ROBOTS_RADIUS+BALL_RADIUS)
            # TODO calcul du tire (pour l'instant on suppose que la balle est mobile)
            robot.hasBall = False
            ball.iscatch = False
            break

    # Historique des variables
    time_last_record = 0
    trajectory_ball = [tuple(ball.pos)] # record tout les time_step_affichage_ms (+ event collision)
    speeds_ball = [np.linalg.norm(ball.velocity)] # record tout les time_step_affichage_ms (+ event collision)
    ennemis_trajectories = {i: [tuple(robot.pos)] for i, robot in enumerate(robots_ennemi)} # record tout les time_step_ms
    ennemis_direction = {i: [tuple(robot.normal)] for i, robot in enumerate(robots_ennemi)} # record tout les time_step_affichage_ms
    ennemis_velocity_records = {i: [tuple(robot.velocity)] for i, robot in enumerate(robots_ennemi)} # record tout les time_step_affichage_ms
    allies_trajectories = {i: [tuple(robot.pos)] for i, robot in enumerate(robots_allie)} # record tout les time_step_ms
    allies_direction = {i: [tuple(robot.normal)] for i, robot in enumerate(robots_allie)} # record tout les time_step_affichage_ms
    allies_velocity_records = {i: [tuple(robot.velocity)] for i, robot in enumerate(robots_allie)} # record tout les time_step_affichage_ms


    ################### ALGO SIMULATION ###################
    DONT_STOP = True
    old_pos_ball = ball.pos
    time = 0 
    TIME_MAX = 3
    while DONT_STOP :
        time_last_record += time_step_ms
        time += dt

        # Update pos ball theorique
        if (not ball.iscatch) :
            old_pos_ball = ball.pos
            ball.update_position(dt)

        # Update pos-vel robots + gestion collision entre robots
        # robots_ennemi[0].update_vel_pos_tocatchball_1(ball.pos, ball.velocity, dt)
        # robots_ennemi[1].update_vel_pos_tocatchball_2(ball.pos, ball.velocity, dt)
        # ennemis_trajectories[0].append(tuple(robots_ennemi[0].pos))
        # ennemis_trajectories[1].append(tuple(robots_ennemi[1].pos))
        # for i, robot in enumerate(robots_ennemi):
        #     dir_to_ball = ball.pos -  robots_ennemi[i].pos
        #     dist_to_ball = np.linalg.norm(dir_to_ball)
        #     if dist_to_ball <= (ROBOTS_RADIUS+BALL_RADIUS):
        #         print(i)
        #         time_last_record = time_step_affichage_ms
        #         DONT_STOP = False

        # Update position robot ennemis 
        for i, robot in enumerate(robots_ennemi) : 
            if (not robot.hasBall):
                # Strategie de d'action-déplacement (pour l'instant everyone attaque : Go catch ball) Avance vers la balle
                # On connait : position_ball - vitesse_ball - position robot - vitesse robot
                # On veux : atteindre la ball avec le même vecteur vitesse en un temps minimum 

                # Version point target 
            
                # Etape 1 : Trouver le point cible (ça sert à rien de courrir vers la balle comme un teubé, si on ne peux pas l'avoir au début)
                    # Optimisation calcul -> Robot stock point cible et actualise seulement si pos théorique ball a changé (collision ou autre)
                    # TODO : list des points atteint par la ball (stock + del un par un)
                    # TODO : calcul temps min pour atteindre un point donné avec vecteur vitesse (ball) (stock)
                    # TODO : calcul temps min pour atteindre un point donné avec vecteur vitesse (robot)
                    # TODO : Prendre le premier point où temps_min ball > temps_min robot 
                
                # On connait : point cible, vitesse cible, pos actuelle, vitesse actuelle,
                # point_tangente_cible, vitesse_tangente_cible, distance (point_tangent + point_cible) (arc), rotation_robot, rotation_target (stocké)
                
                
                ##### distance cible < rayon cercle cible : On recalcul cercle et point tangent et go 




                # Etape 2 : 
                #   Si angle (vitesse_tangente_cible, vitesse_robot) > angle_max :
                #           on tourne dans rotation_robot avec angle_max 
                #           on accelère (sauf si distance(pos_actuelle-point_tangente_cible+ arc) trop petite  )
                #                                   possible : calcule nouveau cercle à v+accel -> point tangent new cercle -> distance arc
                #   Sinon :
                #       si : distance cible > rayon cercle cible # on a quitter le premier cercle mais on est pas au second 
                #           on tourne dans rotation_robot avec angle_max 
                #           on accelère (sauf si distance(pos_actuelle-point_tangente_cible+ arc) trop petite  )
                #       sinon : # on est dans la fin = rotation final pour allignement
                #           TODO
               


                robot.update_vel_pos_tocatchball(ball.pos, ball.velocity, dt)
                # potentiel collision -> nouvelle position balle

                # Orientation normal robot : Tout les robots s'orientent vers la balle 
                direction_to_ball = ball.pos - np.array(robot.pos)
                distance_to_ball = np.linalg.norm(direction_to_ball)
                if distance_to_ball > (BALL_RADIUS + ROBOTS_RADIUS) :
                    robot.normal = direction_to_ball / np.linalg.norm(direction_to_ball) # sinon la normal reste la même
                else : #TODO : change en cas de rebond  
                    ball.iscatch = True
                    robot.hasBall = True 

                    # Stop
                    time_last_record = time_step_affichage_ms
                    DONT_STOP = False

            else :
                # TODO : Modifier la vitesse du robot 

                # TODO : Trouve le point de tir/passe optimal par rapport à sa position actuelle 
                shoot_target = np.array([0, FIELD_DIMENSIONS[1]/2])

                robot.update_direction_vector(shoot_target, dt)
                ball.pos = robot.pos + robot.normal * (ROBOTS_RADIUS + BALL_RADIUS)
                ball.velocity = robot.velocity

            
        # Ajout position actuel
            ennemis_trajectories[i].append(tuple(robot.pos))


        # for robot in zip(robots_ennemi, robots_allie):

        #     if ROBOT(robot).hasBall :
        #         ball.pos = robot.pos + robot.normal * (ROBOTS_RADIUS + BALL_RADIUS)
        #         ball.velocity = robot.velocity
        #         break 
        
        for i, robot in enumerate(robots_allie) : allies_trajectories[i].append(tuple(robot.pos))


        # Gestion collision ball-robot + modif pos-vel ball + modif iscatch 
        # if not BALL(ball).iscatch:
        #     handle_collision_2D
        # Vérification des collisions avec les obstacles #TODO améliorer l'instant de collision
        # for i, obstacle in enumerate(obstacles_pos):
        #     obstacles_trajectories[i].append(tuple(obstacle)) # Enregistrement trajectoire
        #     dist_to_obstacle = np.linalg.norm(position_balle - np.array(obstacle))
        #     if dist_to_obstacle <= (BALL_RADIUS + OBSTACLE_RADIUS) :
        #         velocity = handle_collision(velocity, obstacles_velocities[i], 
        #                                     position_balle, np.array(obstacle), time_step_s)




       
        # Vérification des collisions avec les limites du terrain
        if ((ball.pos[0] - BALL_RADIUS) < 0 or (ball.pos[0] + BALL_RADIUS) > field_width or
            (ball.pos[1] - BALL_RADIUS) < 0 or (ball.pos[1] + BALL_RADIUS) > field_height):
            ball.velocity[0] = ball.velocity[1] = 0

        # if np.linalg.norm(ball.velocity) < 1e-2 or time >= TIME_MAX:
        #     time_last_record = time_step_affichage_ms
        #     DONT_STOP = False

        if time >= TIME_MAX :
            time_last_record = time_step_affichage_ms
            DONT_STOP = False
        # Enregistrement de la trajectoire et de la vitesse
        if time_last_record >= time_step_affichage_ms:
            trajectory_ball.append(tuple(ball.pos)) 
            speeds_ball.append(np.linalg.norm(ball.velocity))
            for i, robot in enumerate(robots_ennemi):
                ennemis_velocity_records[i].append(tuple(robot.velocity))
                ennemis_direction[i].append(tuple(robot.normal))
            for i, robot in enumerate(robots_allie):
                allies_velocity_records[i].append(tuple(robot.velocity))
                allies_direction[i].append(tuple(robot.normal))
            time_last_record = 0 # Reset

    ###### Visualisation #####
    Affichage_Historique(trajectory_ball, speeds_ball, 
                         ennemis_trajectories, ennemis_direction, ennemis_velocity_records, 
                         allies_trajectories, allies_direction, allies_velocity_records, 
                         time_step_affichage_ms, time_step_ms)


simulate_ball_trajectory(ball = ball2, 
                        robots_ennemi=[robot_E1, robot_E2],
                        robots_allie=[robot_A1],
                        time_step_ms= 50,
                        time_step_affichage_ms= 200)

def AffichageTest(pos_robot, pos_target, velocity_robot, velocity_target):
    plt.figure(figsize=(10, 6))
    # FIELD
    plt.gca().add_patch(Rectangle((0, 0), FIELD_DIMENSIONS[0], FIELD_DIMENSIONS[1], fill=True, color="green", linewidth=2)) # Gason
    plt.gca().add_patch(Rectangle((0, (FIELD_DIMENSIONS[1] - GOAL_SIZE) / 2), GOAL_DEPTH, GOAL_SIZE, fill=False, edgecolor="white", linewidth=2))  # Goal Allie - Left side
    plt.gca().add_patch(Rectangle((FIELD_DIMENSIONS[0]- GOAL_DEPTH, (FIELD_DIMENSIONS[1] - GOAL_SIZE) / 2), GOAL_DEPTH, GOAL_SIZE, fill=False, edgecolor="white", linewidth=2)) # Goal Ennemie - Right side

    centre_cercle_robot, rayon_cercle_robot, sens_rotation_robot, centre_cercle_target, rayon_cercle_target, sens_rotation_target = FoundGoodCircle(pos_robot, pos_target, velocity_robot, velocity_target)
    # Cercle
    plt.gca().add_patch(Circle(centre_cercle_robot, rayon_cercle_robot, color="red", fill=True, zorder=2)) # Ajoute position balle intermédiaire
    plt.gca().add_patch(Circle(centre_cercle_target, rayon_cercle_target, color="red", fill=True, zorder=2)) # Ajoute position balle intermédiaire
    plt.quiver(pos_robot[0], pos_robot[1], velocity_robot[0], velocity_robot[1], angles='xy', scale_units='xy', scale=2, color="purple", width=0.003, zorder= 3)
    plt.quiver(pos_target[0], pos_target[1], velocity_target[0] , velocity_target[1], angles='xy', scale_units='xy', scale=2, color="purple", width=0.003, zorder= 3)

    if (sens_rotation_robot < 0 and sens_rotation_target > 0):
        p1, p2 = tangentes_inter_1(centre_cercle_robot[0], centre_cercle_robot[1], rayon_cercle_robot, centre_cercle_target[0], centre_cercle_target[1], rayon_cercle_target)
      
    elif (sens_rotation_robot > 0 and sens_rotation_target < 0):  
        p1, p2 = tangentes_inter_2(centre_cercle_robot[0], centre_cercle_robot[1], rayon_cercle_robot, centre_cercle_target[0], centre_cercle_target[1], rayon_cercle_target)
       
    elif (sens_rotation_robot > 0 and sens_rotation_target > 0):  
        p1, p2 = tangentes_ext_1(centre_cercle_robot[0], centre_cercle_robot[1], rayon_cercle_robot, centre_cercle_target[0], centre_cercle_target[1], rayon_cercle_target)
      
    elif (sens_rotation_robot < 0 and sens_rotation_target < 0):  
        p1, p2 = tangentes_ext_2(centre_cercle_robot[0], centre_cercle_robot[1], rayon_cercle_robot, centre_cercle_target[0], centre_cercle_target[1], rayon_cercle_target)
       
    if (p1 != 0):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="--", color="purple", zorder= 3)

    else :
        # TODO : trajectoire suivit classique 
        a = 0

    # Configuration du graphique
    plt.xlim(-1 , FIELD_DIMENSIONS[0] + 1)
    plt.ylim(-1 , FIELD_DIMENSIONS[1] + 1)
    plt.xlabel("position_balle X (m)")
    plt.ylabel("position_balle Y (m)")
    plt.title("Simulation de la trajectoire d'une balle avec obstacles")
    plt.grid()
    plt.show()


# AffichageTest(pos_robot= np.array([3.5, 2.0]), 
#               pos_target=np.array([5.0, 7.0]),
#               velocity_robot=np.array([0.0, 0.0]),
#               velocity_target=np.array([-1.0, 0.0]))

