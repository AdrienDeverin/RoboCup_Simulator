import math
import matplotlib.pyplot as plt

def compute_speed_profile(
    segment_lengths,   # liste des longueurs de chaque segment: [L_AB, L_BC, L_CD, ...]
    vmax_list,         # liste des vitesses max autorisées pour chaque segment
    v_init,            # vitesse initiale (imposée, ex: V(A))
    v_final,           # vitesse finale (imposée, ex: V(D) = Vmax(segment final))
    a_max,             # accélération maximale
    d_max              # décélération maximale
):
    """
    Calcule le profil de vitesses 'optimal' (temps minimal) au niveau de chaque point
    de jonction (A, B, C, D, ...), puis construit un profil de vitesse détaillé (position, vitesse)
    pour chaque segment.

    Retourne un tuple (node_speeds, profile_points) où :
      - node_speeds = [v(A), v(B), v(C), ...]  (vitesses au points de jonction)
      - profile_points = liste de (x, v) échantillonnant le profil complet le long de la distance.
    """
    # --- supprime segment nul ---
    indices_a_conserver = [i for i in range(len(segment_lengths)) if segment_lengths[i] > 0]
    segment_lengths= [segment_lengths[i] for i in indices_a_conserver]
    vmax_list = [vmax_list[i] for i in indices_a_conserver]

    n = len(segment_lengths)
    if n == 0:
        # Aucun segment...
        if math.isclose(v_init, v_final, rel_tol=1e-9):
            return "No moove"
        else:
            return "Error"

    # speeds[i] sera la vitesse au point i (0 -> A, 1 -> B, 2 -> C, etc.)
    speeds = [0.0] * (n + 1)
    
    # ---------------------
    # 1) Passage en avant
    # ---------------------
    speeds[0] = v_init
    for i in range(1, n+1):
        dist = segment_lengths[i-1]             # distance du segment i-1
        v_prev = speeds[i-1]                    # vitesse au point précédent
        v_possible = math.sqrt(v_prev**2 + 2*a_max*dist) # Vitesse possible via accélération sur dist: v^2 = v_prev^2 + 2*a_max*dist
        # On limite par la vmax du segment
        speeds[i] = min(v_possible, vmax_list[i-1])

    # ---------------------
    # 2) Passage en arrière
    # ---------------------
    if (speeds[n] < v_final):
        return "No solution : trop lent"

    speeds[n] = v_final
    for i in range(n-1, -1, -1):
        dist = segment_lengths[i]
        v_next = speeds[i+1]
        v_possible_back = math.sqrt(v_next**2 + 2*d_max*dist) # Vitesse possible via décélération sur dist: v^2 = v_next^2 + 2*d_max*dist
        # On limite par la vitesse déjà calculée et la vmax du segment
        speeds[i] = min(speeds[i], v_possible_back, vmax_list[i])

    if speeds[0] != v_init : # n'a pas assez d'espace pour décélérer jusqu'a v_final (trop rapide)
        return "No solution : trop rapide"

    # ---------------------
    # Vérifications finales
    # ---------------------
    # Vérif aucune vitesse intermédiaire négative
    for v in speeds:
        if v < 0 or math.isnan(v):
            return "Error"


    # Les "speeds" au points de jonction sont cohérents, on construit le profil détaillé
    profile_points = build_detailed_profile(speeds, segment_lengths, vmax_list, a_max, d_max)

    return speeds, profile_points

def build_detailed_profile(speeds, segment_lengths, vmax_list, a_max, d_max):
    """
    Construit le profil de vitesse détaillé (position, vitesse) en parcourant
    chaque segment selon un schéma trapézoïdal.

    - speeds[i] = vitesse au point i
    - segment_lengths[i] = longueur du segment i
    - vmax_list[i] = vitesse max autorisée sur segment i
    - a_max, d_max = accel / decel max

    Retourne une liste de tuples (x, v) échantillonnés aux points clés.
    x = distance cumulée depuis le début (point A).
    """
    profile_points = []
    x_cumul = 0.0  # position cumulée depuis A

    n = len(segment_lengths)
    # On ajoute le point de départ
    profile_points.append((x_cumul, speeds[0]))

    for i in range(n):
        v_start = speeds[i]
        v_end = speeds[i+1]
        L = segment_lengths[i]
        v_seg_max = vmax_list[i]

        # On calcule un petit profil trapézoïdal (distance 0 -> L).
        # Distances nécessaires pour accélérer de v_start à v_peak :
        #    d_acc = (v_peak^2 - v_start^2) / (2 a_max)   si v_peak > v_start
        # et pour décélérer de v_peak à v_end :
        #    d_dec = (v_peak^2 - v_end^2) / (2 d_max)     si v_peak > v_end
        # On veut v_peak <= v_seg_max.

        # 1) Tenter de monter à v_seg_max si possible
        #    On check la distance nécessaire pour aller de v_start à v_seg_max et redescendre à v_end.
        d_acc_to_vmax = 0.0
        d_dec_from_vmax = 0.0
        can_accelerate = (v_seg_max > v_start + 1e-9)
        can_decelerate = (v_seg_max > v_end + 1e-9)

        if can_accelerate:
            d_acc_to_vmax = (v_seg_max**2 - v_start**2)/(2*a_max)
        if can_decelerate:
            d_dec_from_vmax = (v_seg_max**2 - v_end**2)/(2*d_max)

        # Distance totale pour un "palier" à v_seg_max :
        distance_needed_vmax = d_acc_to_vmax + d_dec_from_vmax

        if distance_needed_vmax <= L and v_seg_max >= v_start and v_seg_max >= v_end:
            # => On peut atteindre v_seg_max, éventuellement tenir un palier, puis redescendre
            v_peak = v_seg_max
            d_cruise = L - distance_needed_vmax
        else:
            # => On ne peut pas atteindre v_seg_max
            # On cherche la v_peak qu'on atteint exactement sur la distance L,
            # en partant à v_start et finissant à v_end.
            # Formule standard : d_total = d_acc + d_dec
            # => (v_peak^2 - v_start^2)/(2*a_max) + (v_peak^2 - v_end^2)/(2*d_max) = L
            # On résout pour v_peak^2 :
            # v_peak^2 * (1/(2*a_max) + 1/(2*d_max)) = L + v_start^2/(2*a_max) + v_end^2/(2*d_max)
            # NB: si v_start > v_end, on décélère plus qu'on accélère, mais la formule reste valable
            #    tant que a_max, d_max > 0.

            num = 2*L + (v_start**2)/a_max + (v_end**2)/d_max
            den = (1/a_max + 1/d_max)
            v_peak_squared = num / den if den != 0 else 0.0
            if v_peak_squared < 0:
                # Profil impossible (pas de solution réelle)
                v_peak = 0
            else:
                v_peak = math.sqrt(max(v_peak_squared, 0))

            # On doit aussi s'assurer de ne pas dépasser v_seg_max
            if v_peak > v_seg_max:
                v_peak = v_seg_max

            d_acc_to_vpeak = max(0.0, (v_peak**2 - v_start**2)/(2*a_max)) if v_peak > v_start else 0.0
            d_dec_from_vpeak = max(0.0, (v_peak**2 - v_end**2)/(2*d_max)) if v_peak > v_end else 0.0

            distance_needed = d_acc_to_vpeak + d_dec_from_vpeak
            d_cruise = L - distance_needed
            if d_cruise < -1e-9:
                # => théoriquement pas possible...
                # On suppose qu'on a déjà vérifié la faisabilité avant (forward-backward).
                # Donc on ne devrait pas arriver ici sauf cas limite numérique.
                # On générera quand même un "profil" en se contentant d'une phase unique
                # d'accélération/décélération (profil triangulaire).
                pass

        # => On a v_peak et d_cruise
        # On va construire les points (position, vitesse) clés :
        seg_profile = build_segment_trapezoid(
            v_start, v_end, L, v_peak, a_max, d_max, x_cumul
        )

        # On ajoute ces points à la liste globale, en évitant de doubler la position de départ
        # (puisqu'on a déjà profile_points[-1])
        # Donc on saute le premier point de seg_profile si c'est la même position que le dernier global.
        if len(seg_profile) > 0:
            # seg_profile[0] devrait correspondre au (x_cumul, v_start)
            # on l'a déjà en profile_points si c'est identique
            for j, (x_p, v_p) in enumerate(seg_profile):
                if j == 0 and math.isclose(x_p, x_cumul, rel_tol=1e-12):
                    # on saute le premier point pour ne pas le dupliquer
                    continue
                profile_points.append((x_p, v_p))

        x_cumul += L  # on met à jour la position de fin de segment

    return profile_points

def build_segment_trapezoid(v_start, v_end, L, v_peak, a_max, d_max, x_offset=0.0):
    """
    Construit la suite de points (x, v) pour un segment de longueur L,
    partant à v_start et finissant à v_end, avec un 'pic' de vitesse v_peak <= v_seg_max.

    On construit jusqu'à 3 phases:
      1) Accélération de v_start à v_peak
      2) Palier (si distance > 0)
      3) Décélération de v_peak à v_end

    On renvoie une liste de points (x, v) en 'distance cumulée' (x_offset + qq_chose).
    Les points clés sont :
      - début
      - fin accélération
      - début décélération
      - fin de segment
    """
    # Pour éviter trop de points, on ne sample qu'aux ruptures de régime.
    # Vous pouvez raffiner en ajoutant plus de subdivisions si vous voulez un profil plus dense.

    if L <= 1e-12:
        # Segment négligeable => un seul point
        return [(x_offset, v_start)]

    points = []
    # point de départ
    points.append((x_offset, v_start))

    # Calcul des distances d'accélération (start->peak) et de décélération (peak->end)
    d_acc = 0.0
    if v_peak > v_start + 1e-9:
        d_acc = (v_peak**2 - v_start**2) / (2 * a_max)

    d_dec = 0.0
    if v_peak > v_end + 1e-9:
        d_dec = (v_peak**2 - v_end**2) / (2 * d_max)

    d_cruise = L - d_acc - d_dec
    if d_cruise < 0:
        # Cas triangulaire pur (pas de palier constant):
        # On recalcule v_peak pour qu'il corresponde pile à la distance L,
        # mais normalement c'est déjà fait en amont.
        # On peut le laisser tel quel, mais d_cruise sera négatif par arrondi.
        d_cruise = 0

    # 1) Fin d'accélération
    x_acc_end = x_offset + d_acc
    if d_acc > 1e-12:
        points.append((x_acc_end, v_peak))

    # 2) Palier
    x_cruise_end = x_acc_end + d_cruise
    if d_cruise > 1e-12:
        # Début du palier = x_acc_end, v = v_peak
        # Fin du palier = x_cruise_end, v = v_peak
        points.append((x_cruise_end, v_peak))

    # 3) Décélération
    if d_dec > 1e-12:
        x_dec_end = x_cruise_end + d_dec
        # A la fin de la décélération, vitesse = v_end
        # On ajoute un point intermédiaire où on finit la décélération
        # (sauf si c'est la même position que la fin de segment, arrondi numérique)
        points.append((x_dec_end, v_end))
    else:
        x_dec_end = x_cruise_end

    # Par sécurité, on s'assure que la fin est bien à x_offset + L
    # (arrondis numériques possibles)
    x_end = x_offset + L
    if not math.isclose(x_dec_end, x_end, rel_tol=1e-9):
        # On met un dernier point pour marquer la fin du segment
        points.append((x_end, v_end))
    else:
        # si le dernier point n'a pas exactement la bonne position,
        # on corrige la position (on peut faire un petit "snap")
        # ou on n'ajoute rien si c'est déjà un isclose
        pass

    return points



if __name__ == "__main__":
    # Exemple d'utilisation
    segment_lengths = [5.0, 7.0, 5.0]  # longueurs AB, BC, CD
    vmax_list = [0.0, 4, 7.0]         # vitesse max sur [A,B], [B,C], [C,D]
    v_init = 0.0                         # impose V(A) = 5
    v_final = 6.5                        # impose V(D) = 8
    a_max = 2.0                          # accélération max
    d_max = 1.0                          # décélération max

    result = compute_speed_profile(segment_lengths, vmax_list, v_init, v_final, a_max, d_max)
  
    if result == "No solution":
        print("No solution")
    else:
        node_speeds, profile_points = result
        print("Vitesse aux points de jonction :", node_speeds)
        print("Profil (x, v) :")
        for (x, v) in profile_points:
            print(f" x={x:5.2f} m,  v={v:5.2f} m/s")


        # --------------------
        # Affichage matplotlib
        # --------------------
        plt.figure()
        X = [p[0] for p in profile_points]
        V = [p[1] for p in profile_points]
        plt.plot(X, V, label="Profil de vitesse")
        plt.xlabel("Position (m)")
        plt.ylabel("Vitesse (m/s)")
        plt.title("Profil de vitesse le long du parcours")
        plt.grid(True)
        plt.legend()
        plt.show()