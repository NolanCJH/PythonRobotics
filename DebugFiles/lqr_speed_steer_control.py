import cubic_spline_planner
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy as np
"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import sys
sys.path.append("../../PathPlanning/CubicSpline/")


# LQR parameter
Q = np.eye(5)
R = np.eye(2)

# parameters
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):

    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = la.eig(A - B @ K)

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e, sp):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    tv = sp[ind]

    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    A[4, 4] = 1.0
    # print(A)

    B = np.zeros((5, 2))
    B[3, 0] = v / L
    B[4, 1] = dt

    K, _, _ = dlqr(A, B, Q, R)

    x = np.zeros((5, 1))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt
    x[4, 0] = v - tv

    ustar = -K @ x

    # calc steering input

    ff = math.atan2(L * k, 1)
    fb = pi_2_pi(ustar[0, 0])

    # calc accel input
    ai = ustar[1, 0]

    delta = ff + fb

    return delta, ind, e, th_e, ai


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05

    #state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)
    state = State(x=25.0, y=14.0, yaw=90.0, v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    e, e_th = 0.0, 0.0

    while T >= time:
        dl, target_ind, e, e_th, ai = lqr_steering_control(
            state, cx, cy, cyaw, ck, e, e_th, speed_profile)

        state = update(state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    for i in range(40):
        speed_profile[-i] = target_speed / (50 - i)
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile

def get_Switch_back_course(dl):
    ax = [24.899999618530273, 24.900005340576172, 24.90001106262207,\
    24.900014877319336, 24.900020599365234, 24.900026321411133,\
    24.90003204345703, 24.89870262145996, 24.889429092407227,\
    24.864276885986328, 24.815370559692383, 24.735021591186523,\
    24.615943908691406, 24.29094886779785, 23.830982208251953,\
    23.252336502075195, 22.575511932373047, 21.824485778808594,\
    21.385210037231445, 20.937171936035156, 20.484573364257812,\
    20.03000259399414, 19.574806213378906, 19.1195068359375,\
    18.233142852783203, 17.346778869628906, 16.46041488647461,\
    15.574051856994629, 14.687687873840332, 13.801324844360352,\
    12.914960861206055, 12.028596878051758, 11.142233848571777,\
    10.25586986541748, 9.369505882263184, 8.93588638305664,\
    8.50233268737793, 8.069173812866211, 7.637269020080566,\
    7.208261489868164, 6.784831523895264, 6.3709259033203125,\
    6.020498275756836, 5.684054851531982, 5.282763481140137,\
    4.910910606384277, 4.570590019226074, 4.262977123260498,\
    3.9884414672851562, 3.7466609477996826, 3.5367300510406494,\
    3.357264518737793, 3.2064950466156006, 3.082350254058838,\
    2.9825286865234375, 2.9045591354370117, 2.8458502292633057,\
    2.803729772567749, 2.775472640991211, 2.75832462310791,\
    2.7495148181915283, 2.7462656497955322, 2.7457969188690186,\
    2.745792865753174, 2.745788812637329, 2.7457847595214844,\
    2.747112989425659, 2.7563867568969727, 2.7815396785736084,\
    2.8304450511932373, 2.910794734954834, 3.0298728942871094,\
    3.354867458343506, 3.8148345947265625, 4.393479824066162,\
    5.0703043937683105, 5.82133150100708, 6.260606288909912,\
    6.708644866943359, 7.161242961883545, 7.615814685821533,\
    8.07101058959961, 8.526309967041016, 9.351309776306152,\
    10.176310539245605, 11.001310348510742, 11.826310157775879,\
    12.651309967041016, 13.476309776306152, 14.411310195922852,\
    15.34631061553955, 16.28131103515625, 17.216310501098633,\
    18.151309967041016, 19.0863094329834, 20.021310806274414,\
    20.956310272216797, 21.89130973815918, 22.826311111450195]
    ay = [14.199999809265137, 15.033333778381348, 15.866666793823242,\
    16.700000762939453, 17.53333282470703, 18.366666793823242,\
    19.200000762939453, 19.65530014038086, 20.110496520996094,\
    20.565067291259766, 21.01766586303711, 21.46570587158203,\
    21.90498161315918, 22.65601348876953, 23.332843780517578,\
    23.911497116088867, 24.371471405029297, 24.696475982666016,\
    24.81555938720703, 24.895915985107422, 24.944826126098633,\
    24.969985961914062, 24.979265213012695, 24.98059844970703,\
    24.98060417175293, 24.980609893798828, 24.980615615844727,\
    24.980621337890625, 24.980627059936523, 24.980632781982422,\
    24.98063850402832, 24.98064422607422, 24.980648040771484,\
    24.980653762817383, 24.98065948486328, 24.97962760925293,\
    24.9724178314209, 24.95285987854004, 24.914812088012695,\
    24.852230072021484, 24.75927734375, 24.630525588989258,\
    24.485376358032227, 24.310237884521484, 24.048620223999023,\
    23.74658203125, 23.409381866455078, 23.042070388793945,\
    22.649396896362305, 22.235727310180664, 21.80500030517578,\
    21.36069679260254, 20.905839920043945, 20.442995071411133,\
    19.974287033081055, 19.50144386291504, 19.025814056396484,\
    18.54842185974121, 18.070003509521484, 17.59105110168457,\
    17.11186981201172, 16.63261604309082, 16.153350830078125,\
    15.52001667022705, 14.886683464050293, 14.253350257873535,\
    13.798049926757812, 13.342854499816895, 12.88828182220459,\
    12.435683250427246, 11.98764419555664, 11.548367500305176,\
    10.79733657836914, 10.120506286621094, 9.541853904724121,\
    9.081877708435059, 8.756874084472656, 8.637789726257324,\
    8.557435035705566, 8.508523941040039, 8.483365058898926,\
    8.474085807800293, 8.47275161743164, 8.472745895385742,\
    8.47274112701416, 8.472735404968262, 8.47273063659668,\
    8.472725868225098, 8.4727201461792, 8.4727144241333,\
    8.472708702087402, 8.472702026367188, 8.472696304321289,\
    8.47269058227539, 8.472684860229492, 8.472679138183594,\
    8.472672462463379, 8.47266674041748, 8.472661018371582]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    ax = [22.826311111450195, 22.418312072753906, 22.010366439819336,\
    21.602720260620117, 21.196029663085938, 20.791549682617188,\
    20.39132308959961, 19.998374938964844, 19.213205337524414,\
    18.5056095123291, 17.90065574645996, 17.419771194458008,\
    17.079994201660156, 16.961217880249023, 16.878517150878906,\
    16.82509994506836, 16.793806076049805, 16.77730941772461,\
    16.768220901489258, 16.762981414794922, 16.75774383544922]
    ay = [8.472661018371582, 8.471787452697754, 8.465682029724121,\
    8.449118614196777, 8.416891098022461, 8.363858222961426,\
    8.285024642944336, 8.175670623779297, 7.835904121398926,\
    7.355029106140137, 6.750082015991211, 6.042492389678955,\
    5.257328033447266, 4.823906421661377, 4.38212776184082,\
    3.9358205795288086, 3.4873881340026855, 3.0381479263305664,\
    2.58868670463562, 2.2887322902679443, 1.9887781143188477]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    s.extend(s2)
    return cx, cy, cyaw, ck, s

def main():
    print("LQR steering control tracking start!!")
    #ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    #ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    #goal = [ax[-1], ay[-1]]
    #cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
    #    ax, ay, ds=0.1)

    dl = 1.0  # course tick
    cx, cy, cyaw, ck, s = get_Switch_back_course(dl)
    target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s
    goal = [cx[-1], cy[-1]]
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)

    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()


if __name__ == '__main__':
    main()
