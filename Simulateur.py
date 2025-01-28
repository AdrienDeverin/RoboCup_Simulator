from constantes_physique import * # include np
from utils import *
from Motion_function import *
from objects import BALL, ROBOT


initial_speed= 5 # m.s-1
angle_rad = np.radians(45) # 45°
ball = BALL(pos = np.array([1.0, 1.0]), 
            velocity= np.array([initial_speed * np.cos(angle_rad), initial_speed * np.sin(angle_rad)], dtype=float), 
            iscatch=False)

ball2 = BALL(pos = np.array([10.0, 12.0]), 
            velocity= np.array([1,0], dtype=float), 
            iscatch=False)

robot_A1 = ROBOT(pos = np.array([1.0, 1.0]), 
                velocity = np.array([0 , 0]), 
                normal = np.array([1.0, 1.0]), 
                isON = True, hasBall = True)

robot_E1 = ROBOT(pos = np.array([10.0, 2.0]), 
                velocity = np.array([-3.5 , 0]), 
                normal = np.array([-2.0, 0.0]), 
                isON = True, hasBall = False)


# robot_E2 = ROBOT(pos = np.array([8.0, 6.0]), 
#                 velocity = np.array([0 , 2]), 
#                 normal = np.array([-2.0, 0.0]), 
#                 isON = True, hasBall = False)

def simulate_ball_trajectory(ball, robots_ennemi, robots_allie, time_step_ms=50, time_step_affichage_ms=200):
    
    # Dimensions limite affichage
    field_width, field_height = FIELD_DIMENSIONS
    dt = time_step_ms / 1000  # Conversion ms -> s

    # tir initial, forcé pour le test
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
    TIME_MAX = 5

    while DONT_STOP :
        time_last_record += time_step_ms
        time += dt

        # Update pos ball theorique
        # if (not ball.iscatch) :
        #     old_pos_ball = ball.pos
        #     ball.update_position(dt) # TODO : prendre en compte tout les obstacle et gérer rebond

        # Update position robot ennemis 
        for i, robot in enumerate(robots_ennemi) : 
            if (not robot.hasBall):
                # Strategie de d'action-déplacement (pour l'instant everyone attaque : Go catch ball) Avance vers la balle
                # TODO : Verifier si un allie à la ball déjà ... 
                ACTION_DESISION = "CATCH_BALL"

                if (ACTION_DESISION == "CATCH_BALL"):

                    ####### Part 1 : Trouver cible + Déplacement (si possible) ######
                    # On connait : position_ball - vitesse_ball - position robot - vitesse robot
                    # On veux : atteindre la ball avec le même vecteur vitesse en un temps minimum 
                    #####################
                    if (robot.targetPoint_isupdated): 
                        # Determine target point 
                        ball_speed = np.linalg.norm(ball.velocity) 
                        if (ball_speed > 1e-5):
                            direction_to_ball = ball.velocity/ball_speed
                        else :
                            direction_to_ball = -(ball.pos - np.array(robot.pos))
                            direction_to_ball /= np.linalg.norm(direction_to_ball)

                        epsilon = 0.01
                        target_point = ball.pos + direction_to_ball*(ROBOTS_RADIUS+BALL_RADIUS-epsilon)

                        Accel, Angle, Time, pt_cible, v_cible = Trajectory_Planner(robot.pos, robot.velocity, target_point, ball.velocity, dt)
                        robot.Accel = Accel
                        robot.Angle = Angle
                        robot.Time = Time
                        robot.TrajectoryStepPoint = pt_cible
                        robot.TrajectoryStepVel = v_cible
                        robot.targetPoint = target_point
                        robot.targetVelocity = ball.velocity
                        robot.targetPoint_isupdated = False 

                    if (len(robot.Time)> 0):
                        # Avancer selon trajectoire 
                        # TODO : AJOUTER esquiver ennemis ou limiter distance contacte et limite terrain (prendre liste de tous les robots sauf celui ci)
                        
                        if (Time[0] > dt or len(Time) == 1):
                            speed = np.linalg.norm(robot.velocity)
                            if (speed > 1e-5): 
                                new_dir = rotate_vector(robot.velocity, robot.Angle[0]*dt)/speed
                            else :
                                new_dir = robot.TrajectoryStepPoint[0] - robot.pos
                                new_dir /= np.linalg.norm(new_dir)

                            new_speed = min(speed + robot.Accel[0]*dt, ROBOTS_MAX_SPEED) # comparaison unecessary ? 
                            if (new_speed < 0):
                                new_speed = 0
                                robot.targetPoint_isupdated = True
                                DONT_STOP = False

                            # Update velocity
                            robot.velocity = new_speed * new_dir
                            # Update pos 
                            robot.pos = robot.pos + robot.velocity * dt
                            robot.Time[0] -= dt

                        else :
                            Time_step_1 = Time[0]
                            Time_step_2 = min(dt-Time_step_1, Time[1])
                            Time[1] -= Time_step_2

                            speed = np.linalg.norm(robot.velocity)
                            if (speed > 1e-5): 
                                new_dir = rotate_vector(robot.velocity, robot.Angle[0]*(Time_step_1)) 
                                new_dir /= speed
                                new_dir + rotate_vector(new_dir, robot.Angle[1]*(Time_step_2))
                            else :
                                new_dir = robot.TrajectoryStepPoint[1] - robot.pos
                                new_dir /= np.linalg.norm(new_dir)

                            new_speed = min(speed + robot.Accel[0]*Time_step_1 + robot.Accel[1]*Time_step_2, ROBOTS_MAX_SPEED) 
                            if (new_speed < 0):
                                new_speed = 0
                                robot.targetPoint_isupdated = True
                                DONT_STOP = False

                            # Update velocity
                            robot.velocity = new_speed * new_dir
                            # Update pos 
                            robot.pos = robot.pos + robot.velocity * dt
                            robot.Time[0] -= Time_step_1
 
                        while (len(robot.Time)> 0 and robot.Time[0] <= 0 ):
                            del robot.Time[0]
                            del robot.Accel[0]
                            del robot.Angle[0]
                            del robot.TrajectoryStepPoint[0]
                            del robot.TrajectoryStepVel[0]

                    else :
                        print ("Fin")
                        robot.targetPoint_isupdated = True



                    # current_speed =  np.linalg.norm(robot.velocity)
                    # # Si on est trop proche de la cible (distance cible < rayon cercle cible) : suivit classique 
                    # if (np.linalg.norm(ball.pos - np.array(robot.pos)) < (max(current_speed, np.linalg.norm(ball.velocity))/ROBOTS_MAX_ANGULAR_SPEED + ROBOTS_RADIUS + BALL_RADIUS)) :
                    #     robot.update_vel_pos_tocatchball(ball.pos, ball.velocity, dt) # TODO : esquiver ennemis ou limiter distance contacte et limite terrain (prendre liste de tous les robots sauf celui ci) 

                    # else : 
                    #     # Optimisation calcul -> Robot stock point cible et actualise seulement angle valocity à changé (#TODO plutôt si pos théorique ball a changé ?) (collision ou erreur vitesse initiale)
                    #     if (robot.targetPoint is None or robot.targetVelocity is None or angle_between_vectors(robot.targetVelocity, ball.velocity) > 1e-5): # point change :
                    #         list_futur_pos, list_futur_vel, list_futur_time_ball = ball.futur_points(dt)
                    #         time_to_catch, distance_trajet, dir_tangent, sens_rotation = robot.update_targetPoint(list_futur_pos, list_futur_vel, list_futur_time_ball, dt) # Trouve le point cible et donne le temps

                    #     # else : # le point est le même 
                    #     #     time_to_catch, distance_trajet, dir_tangent, sens_rotation = calculate_time_and_target_tangent(robot.pos, robot.targetPoint, robot.velocity, robot.targetVelocity)
                        
                    #     if (time_to_catch != None):
                    #         print(f"Estimation time before catching ball = {round(time_to_catch)}s")
                    #         # Avancer selon trajectoire 
                    #         # TODO : AJOUTER esquiver ennemis ou limiter distance contacte et limite terrain (prendre liste de tous les robots sauf celui ci)
                            
                    #         # Réglage angle
                    #         if (current_speed > 1e-5):
                    #             angle_to_target = angle_between_vectors(robot.velocity, dir_tangent)
                    #             max_angle = ROBOTS_MAX_ANGULAR_SPEED * dt 
                    #             new_direction = rotate_vector(robot.velocity, sens_rotation * min(max_angle, angle_to_target))
                    #             new_direction /= np.linalg.norm(new_direction)
                    #         else :
                    #             new_direction = dir_tangent

                    #         # Réglage speed # Ralentir si on s'éloigne ? ...
                    #         max_accel = ACCELERATION_RATE *dt
                    #         max_decel = DECELERATION_RATE *dt
                    #         epsilon = 0.03 # volume catch ball
                    #         if (current_speed + max_accel > robot.targetSpeed): 
                    #             distance_to_stop = (current_speed*current_speed - robot.targetSpeed*robot.targetSpeed)/(2*DECELERATION_RATE) + ROBOTS_RADIUS + BALL_RADIUS - epsilon # ditance à cette vitesse
                    #             if (distance_to_stop >= distance_trajet ) : #ON va trop vite -> freine 
                    #                 speed_new = max(current_speed - max_decel, robot.targetSpeed)
                    #             else : 
                    #                 d_supp = (2*current_speed*max_accel + max_accel*max_accel)/(2*DECELERATION_RATE)
                    #                 if ((distance_to_stop+ d_supp) > distance_trajet):
                    #                     # maintiend vitesse courante
                    #                     speed_new = current_speed
                    #                 else :
                    #                     # accel
                    #                     speed_new = min(current_speed + max_accel, ROBOTS_MAX_SPEED)
                    #         else :
                    #             distance_to_stop = ((current_speed + max_accel)*(current_speed+ max_accel) - robot.targetSpeed*robot.targetSpeed)/(2*DECELERATION_RATE) + ROBOTS_RADIUS + BALL_RADIUS - epsilon
                    #             if (distance_to_stop >= distance_trajet ) : # On accelère 
                    #                 speed_new = min(current_speed + max_accel, ROBOTS_MAX_SPEED)
                    #             else : 
                    #                 speed_new = robot.targetSpeed

                    #         # Update velocity
                    #         robot.velocity = speed_new * new_direction
                    #         # Update pos 
                    #         robot.pos = robot.pos + robot.velocity * dt

                    #     else :
                    #         print("Interception Impossible !")

                    ###### Part 2 : Gestion Catch bell #####
                    # Update position/iscatch balle
                    ########################################
                    direction_to_ball = ball.pos - np.array(robot.pos)
                    distance_to_ball = np.linalg.norm(direction_to_ball) # nouvelle distance
                    if (distance_to_ball < (ROBOTS_RADIUS + BALL_RADIUS)) :
                        delta_speed = 0 # TODO
                        if (delta_speed > GRIPPER_ABSORPTION) : # TODO ou mal orienté : REBOND
                            ball.velocity = handle_collision_2D(ball.velocity, robot.velocity, ball.pos, robot.pos) # TODO : Calculer point d'intersection (graçe à old_pos_ball )
                        else : # BALL is catch
                            robot.hasBall = True
                            if not ball.iscatch:
                                ball.iscatch = True
                            else : # on la retire à celui qui l'avait
                                for i, robot_tested in enumerate(zip(robots_ennemi, robots_allie)):
                                    if (robot_tested.hasball):
                                        robot_tested.hasball = False
                                        break

                            # Update ball pos, vel with the robot
                            ball.pos = robot.pos + robot.normal * (ROBOTS_RADIUS + BALL_RADIUS)
                            ball.velocity = robot.velocity

                        # Stop
                        time_last_record = time_step_affichage_ms
                        DONT_STOP = False
                          
                    ######## Part 3 : Orientation direction préhenseur robot 
                    #  Tout les robots s'orientent vers la balle 
                    ########################################################
                    if distance_to_ball > ROBOTS_RADIUS: # normalement toujours vrai si instruction précédente correct
                        robot.normal = direction_to_ball / np.linalg.norm(direction_to_ball) # sinon la normal reste la même

                else :
                    continue # Autre scenario non résolu

            else :
                # TODO : Trouve le point de tir/passe optimal par rapport à sa position actuelle 
                shoot_target = np.array([0, FIELD_DIMENSIONS[1]/2])
                robot.update_direction_vector(shoot_target, dt)

                # TODO : Update velocity et pos vers ce point (on accelère toujours si possible ?)
                
                # Si l'angle entre velocity,target ou direction, target et trop grand
                ball.pos = robot.pos + robot.normal * (ROBOTS_RADIUS + BALL_RADIUS)
                ball.velocity = robot.velocity

                # Sinon : TODO SHOOT 
                
            
            # Recording current position 
            ennemis_trajectories[i].append(tuple(robot.pos))


        
        for i, robot in enumerate(robots_allie) : 
            # TODO 
            allies_trajectories[i].append(tuple(robot.pos))

       
        # Condition d'arrêt du test
        if ((ball.pos[0] - BALL_RADIUS) < 0 or (ball.pos[0] + BALL_RADIUS) > field_width or
            (ball.pos[1] - BALL_RADIUS) < 0 or (ball.pos[1] + BALL_RADIUS) > field_height):
            ball.velocity[0] = ball.velocity[1] = 0
            time_last_record = time_step_affichage_ms
            DONT_STOP = False
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

    ###### FIN : Visualisation #####
    Affichage_Historique(trajectory_ball, speeds_ball, 
                         ennemis_trajectories, ennemis_direction, ennemis_velocity_records, 
                         allies_trajectories, allies_direction, allies_velocity_records, 
                         time_step_affichage_ms, time_step_ms)


simulate_ball_trajectory(ball = ball2, 
                        robots_ennemi=[robot_E1],
                        robots_allie=[robot_A1],
                        time_step_ms= 10,
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

