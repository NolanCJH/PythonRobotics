"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


k = 0.5  # control gain
Kp = 1.0  # speed propotional gain
dt = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
max_steer = np.radians(30.0)  # [rad] max steering angle

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
   # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = [np.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
    closest_error = min(d)
    target_idx = d.index(closest_error)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      - np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    #ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    #ay = [0.0, 0.0, -30.0, -20.0, 0.0]
    #ax = [0.0, 6.0, 12.5, 5.0, 7.5, 3.0, -1.0]
    #ay = [0.0, 0.0, 5.0, 6.5, 3.0, 5.0, -2.0]

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
        ax, ay, ds=0.1)

    target_speed = 30.0 / 3.6  # [m/s]

    max_simulation_time = 100.0

    # Initial state
    #state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)
    state = State(x=25.0, y=15.0, yaw=np.radians(90.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)

    while max_simulation_time >= time and last_idx > target_idx:
        ai = pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
        state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
