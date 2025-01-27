from Motion_function import *

class ROBOT():
    def __init__(self, pos, velocity, normal, isON = True, hasBall = False):
        """
        GENERE UN ROBOT 
        normal : vecteur de l'orientation du robot (face des préhenseurs)
        """
        if (pos[0] < 0 or pos[0] > FIELD_DIMENSIONS[0] or pos[1] < 0 or pos[1] > FIELD_DIMENSIONS[1]):
            print("/!\ Position outside field dimensions")
            self.pos= np.array([0,0])
        else: self.pos= pos
        if (np.linalg.norm(velocity)> ROBOTS_MAX_SPEED) : 
            print("/!\ Speed initialisation is too hight ! ")
            self.velocity = velocity / np.linalg.norm(velocity) * ROBOTS_MAX_SPEED
        else : self.velocity= velocity
        if (np.linalg.norm(normal) != 1) : normal /= np.linalg.norm(normal)
        self.normal= normal
        self.isON = isON
        self.hasBall = False

        self.targetPoint_isupdated = True
        self.targetPoint = None
        self.targetVelocity = None

        # Trajectory plan
        self.Accel = []
        self.Angle = []
        self.Time = []
        self.TrajectoyStepPoint = []
        self.TrajectoyStepVel = []

        self.targetSpeed = None # to suppr





    def update_direction_vector(self, target_position, dt):
        """
        Update self.normal de tel sorte à être orienté vers une position cible
        """
        target_vector = normalize(target_position - self.pos)
        angle_to_target = angle_between_vectors(self.normal, target_vector)
        max_angle = ROBOTS_MAX_ANGULAR_SPEED * dt # /!\ angle de direction , pas de rotation pur

        if angle_to_target <= max_angle: # Si l'angle est inférieur à l'angle max, on prend directement le vecteur cible
            self.normal = target_vector 
        else:  # Sinon, on tourne le vecteur actuel vers le vecteur cible par l'angle max
            direction_sign = np.sign(np.cross(self.normal, target_vector))  # Sens de rotation (-1 ou +1)
            self.normal = rotate_vector(self.normal, direction_sign * max_angle)
           
    def update_vel_pos_tocatchball(self, p_ball: np.ndarray,     # Position courante de la balle
                                    v_ball: np.ndarray,     # Vitesse courante de la balle
                                    dt: float               # Pas de temps
                                    ):
        """
        Met à jour la position et la vitesse de l'agent à partir des
        contraintes cinématiques et de l'objectif d'interception
        (réduire la distance à la balle + tendre vers la vitesse de la balle).
        """
        if self.hasBall:
            print("Error : robot has already ball !")
            return
        
        # Direction désirée : 
        dir_to_ball = p_ball - self.pos
        dist_to_ball = np.linalg.norm(dir_to_ball) 
        if dist_to_ball > 0:
            dir_to_ball = dir_to_ball / dist_to_ball
        else:
            dir_to_ball = np.zeros_like(dir_to_ball)
            return 
            print("Error direction ball (collision en cours)")
        
        # On calcule la direction actuelle de l'agent (à partir de v_agent)
        speed_agent = np.linalg.norm(self.velocity)
        
        if speed_agent > 1e-8:
            dir_agent = self.velocity / speed_agent
        else:
            # Si l'agent est à l'arrêt, on définit une direction de référence (par ex. dir_to_ball)
            dir_agent = dir_to_ball
        
        norm_dir_des = np.linalg.norm(dir_to_ball)
        if norm_dir_des > 1e-8:
            dir_to_ball = dir_to_ball / norm_dir_des
        else:
            dir_to_ball = dir_agent  # par défaut
        
        # Poderation entre dir_ball et velocity_ball
        speed_target = np.linalg.norm(v_ball)
        alpha = 0.7
        if speed_target > 1e-8:
            dir_target = v_ball / speed_target
            dir_cible = alpha*dir_to_ball + (1-alpha)*dir_target
        else : 
            dir_cible = dir_to_ball


        # Limiter la rotation : on ne peut pas passer instantanément de dir_agent à dir_cible 
        dot_product = np.clip(np.dot(dir_agent, dir_cible), -1.0, 1.0)
        angle = np.arccos(dot_product)  # angle entre les deux directions 

        max_angle = ROBOTS_MAX_ANGULAR_SPEED * dt
        max_accel = ACCELERATION_RATE *dt
        max_decel = DECELERATION_RATE *dt
      
        if angle > max_angle:
            # On tourne seulement de max_angle vers dir_des (+ ralenti ou accelaire en fonction de la distance)

            # signe de la rotation via produit en croix (2D -> z)
            cross_2d = dir_agent[0] * dir_cible[1] - dir_agent[1] * dir_cible[0]
            sign = np.sign(cross_2d)  # +1 ou -1
            # On crée une matrice de rotation en 2D
            cos_a = np.cos(max_angle)
            sin_a = np.sin(max_angle * sign)  # signe du pivot
            # On utilise sign pour savoir le sens de rotation
            new_dir_x = dir_agent[0] * cos_a - dir_agent[1] * sin_a
            new_dir_y = dir_agent[0] * sin_a + dir_agent[1] * cos_a
            dir_agent_new = np.array([new_dir_x, new_dir_y])

            
        else:
            dir_agent_new = dir_cible # On peut s'orienter directement vers dir_des

        # Calcul vitesse # Angle n'a pas d'impact -> on accèlere si possible
        dist_to_ball = np.linalg.norm(p_ball - (self.pos + dir_agent*speed_agent*dt) ) # distance à t+1 si on maintien la speed
        epsilon = 0.03 # volume catch ball (évite d'osciller lorsque très proche de la cible) 3cm

        if (speed_agent+max_accel > speed_target): 
            distance_to_stop = (speed_agent*speed_agent - speed_target*speed_target)/(2*DECELERATION_RATE) + ROBOTS_RADIUS + BALL_RADIUS - epsilon # ditance à cette vitesse
            if (distance_to_stop >= dist_to_ball ) : #ON va trop vite -> freine 
                speed_new = max(speed_agent - max_decel, speed_target)
            else : 
                d_supp = (2*speed_agent*max_accel + max_accel*max_accel)/(2*DECELERATION_RATE)
                if ((distance_to_stop+ d_supp) > dist_to_ball):
                    # maintiend vitesse courante
                    speed_new = speed_agent
                else :
                    # accel
                    speed_new = min(speed_agent + max_accel, ROBOTS_MAX_SPEED)
        else :
            distance_to_stop = ((speed_agent + max_accel)*(speed_agent + max_accel) - speed_target*speed_target)/(2*DECELERATION_RATE) + ROBOTS_RADIUS + BALL_RADIUS - epsilon
            if (distance_to_stop >= dist_to_ball ) : # On accelère 
                speed_new = min(speed_agent + max_accel, ROBOTS_MAX_SPEED)
            else : 
                speed_new = speed_target


        # Nouvelle vitesse vectorielle
        self.velocity = dir_agent_new * speed_new
        # Mise à jour de la position
        self.pos = self.pos + self.velocity * dt

    def update_targetPoint(self, list_futur_pos, list_futur_vel, list_futur_time_ball, dt):
        for i in range(len(list_futur_pos)):
            time, distance, dir_tangent, sens_rotation = calculate_time_and_target_tangent(self.pos, list_futur_pos[i], self.velocity, list_futur_vel[i])
            if (time != None) :
                if (time + dt < list_futur_time_ball[i]):
                    # Update target point 
                    self.targetPoint = list_futur_pos[i]
                    self.targetVelocity = list_futur_vel[i]
                    self.targetSpeed = np.linalg.norm(list_futur_vel[i])
                    return time, distance, dir_tangent, sens_rotation
        return None, None, None, None
    



class BALL():
    def __init__(self, pos, velocity, iscatch):
        self.pos = pos
        self.velocity = velocity
        self.iscatch = iscatch
    
    def display_info(self):
        print(f"Position: {self.pos}\nVelocity: {self.velocity}\nSpeed : {np.linalg.norm(self.velocity)}\nIs Caught: {self.iscatch}")

    def update_position(self, dt):
        """Update the position of the ball based on its velocity and the time step (dt)."""
        if not self.iscatch:
            self.pos, self.velocity = update_ball_speed(self.pos, self.velocity, dt)
    
    def futur_points(self, dt): 
        list_futur_pos = []
        list_futur_vel =[] 
        list_futur_time =[]
        STOP = False
        pos = self.pos
        vel = self.velocity
        time = 0
        while not STOP : 
            new_pos, new_vel = update_ball_speed(pos, vel, dt)
            time += dt
            if (new_pos[0] < 0 or new_pos[0] > FIELD_DIMENSIONS[0] or new_pos[1] < 0 or new_pos[1] > FIELD_DIMENSIONS[1]): 
                STOP = True
            else : 
                if (np.linalg.norm(new_vel) < 1e-8): 
                    time = np.inf
                    STOP = True
                list_futur_pos.append(new_pos)
                list_futur_vel.append(new_vel)
                list_futur_time.append(time)
                pos = new_pos
                vel = new_vel

        return list_futur_pos, list_futur_vel, list_futur_time
            



