import cv2
import os
import rclpy
import math
import mediapipe as mp
import time
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
import sounddevice as sd
import threading
from scipy.fft import fft
from scipy.signal import butter, lfilter

# Coordinate system
cor_x = 0
cor_y = 0


"""
THIS IS THE MICROPHONE, SOUND AND SIGNAL PART
"""
pause_processing = False
sample_rate = 5120
frame_size = 512
array_length = 5
to_check = 3
stopsending = False
stopsendingconfirm = False
setdistance = True

# confirmflags:
distanceflag = False
stayonpointerflag = False  ## send distance and wait
angleflag = False


def playconfirm():
    global sentvalues, stopsendingconfirm
    if stopsending and stayonpointerflag and not stopsendingconfirm:
        return play_tones("F", "0")
    elif stopsending and distanceflag and not stopsendingconfirm:
        return play_tones("F", "5")
    elif stopsending and angleflag and not stopsendingconfirm:
        return play_tones("F", "A")


def repeated_confirm():
    global stopsendingconfirm, Senddistance
    if stopsendingconfirm:
        Senddistance = False
        return  # Stop sending if the flag is set to True
    else:
        playconfirm()

    # Schedule this function to be called again after the specified delay
    threading.Timer(1, repeated_confirm).start() 


class RollingArray:
    def __init__(self, size=array_length):
        self.size = size
        self.data = []

    def append(self, value):
        if len(self.data) == self.size:
            self.data.pop(0)
        self.data.append(value)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return str(self.data)


ra = RollingArray()
for _ in range(array_length):
    ra.append("G")


class ControlArray:
    def __init__(self, size=array_length):
        self.size = size
        self.data = []

    def append(self, value):
        if len(self.data) == self.size:
            self.data.pop(0)
        self.data.append(value)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return str(self.data)


ca = ControlArray()
for _ in range(array_length):
    ca.append("G")


def find_triplicate(arr):
    if len(arr) != array_length:
        return "Invalid array size"

    frequency_count = {}

    for ele in arr:
        if ele != "G":
            frequency_count[ele] = frequency_count.get(ele, 0) + 1

    # Filter the dictionary to get items with count >= 5
    candidates = {
        key: value for key, value in frequency_count.items() if value >= to_check
    }

    if not candidates:
        return None  # or "No element appears 3 times"

    # Return the element with the highest count
    return max(candidates, key=candidates.get)


def name_of_matched_pair(value_pairs, array):
    matches = []
    for name, values in value_pairs.items():
        if values[0] in array and values[1] in array:
            matches.append(name)  # Collect names of all matched pairs
    if len(matches) == 1:
        return matches[
            0
        ]  # Return the name of the matched pair if only one match is found
    return "G"  # Return "G" if no match or multiple matches are found


value_pairs_r = {
    "0": (700, 1210),
    "1": (700, 1340),
    "2": (700, 1480),
    "3": (700, 1630),
    "4": (770, 1210),
    "5": (770, 1340),
    "6": (770, 1480),
    "7": (770, 1630),
    "8": (850, 1210),
    "9": (850, 1340),
    "A": (850, 1480),
    "B": (850, 1630),
    "C": (940, 1210),
    "D": (940, 1340),
   "E": (940, 1480),
    "F": (940, 1630),
}

control_pairs_r = {
    "0": (400, 1700),
    "1": (400, 1770),
    "2": (400, 1840),
    "3": (400, 1910),
    "4": (470, 1700),
    "5": (470, 1770),
    "6": (470, 1840),
    "7": (470, 1910),
    "8": (550, 1700),
    "9": (550, 1770),
    "A": (550, 1840),
    "B": (550, 1910),
    "C": (620, 1700),
    "D": (620, 1770),
    "E": (620, 1840),
    "F": (620, 1910),
}

value_pairs_p = {
    "A": (700, 1210),
    "B": (700, 1340),
    "8": (700, 1480),
    "9": (700, 1630),
    "E": (770, 1210),
    "F": (770, 1340),
    "C": (770, 1480),
    "D": (770, 1630),
    "2": (850, 1210),
    "3": (850, 1340),
    "0": (850, 1480),
    "1": (850, 1630),
    "6": (940, 1210),
    "7": (940, 1340),
    "4": (940, 1480),
    "5": (940, 1630),
}

control_pairs_p = {
    "A": (400, 1700),
    "B": (400, 1770),
    "8": (400, 1840),
    "9": (400, 1910),
    "E": (470, 1700),
    "F": (470, 1770),
    "C": (470, 1840),
    "D": (470, 1910),
    "2": (550, 1700),
    "3": (550, 1770),
    "0": (550, 1840),
    "1": (550, 1910),
    "6": (620, 1700),
    "7": (620, 1770),
    "4": (620, 1840),
    "5": (620, 1910),
}

indices_before_numpy = [
    40,
    47,
    55,
    62,
    70,
    77,
    85,
    94,
    121,
    134,
    148,
    163,
    170,
    177,
    184,
    191,
]  # extend for purposeful noise inclusion


def resume_processing():
    global pause_processing
    pause_processing = False
    print("Resuming processing...")


def play_tones(key1, key2, duration=0.5, buffer_size=4096, sample_rate=40960):
    # Retrieve frequencies from the dictionaries
    global sentvalues, pause_processing, stopsendingconfirm
    freq1 = control_pairs_r.get(key1)
    freq2 = value_pairs_r.get(key2)
    sentvalues.append(key1)
    sentvalues.append(key2)
    print("Sending first 2 in array: ", sentvalues)

    # If one of the keys is not found, return an error message
    if not freq1 or not freq2:
        print("Invalid key(s)")
        return

    # Combine frequencies
    frequencies = [freq1, freq2]

    # Generate waveform
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)
    for freq_pair in frequencies:
        for f in freq_pair:
            waveform += np.sin(2 * np.pi * f * t)

    # Play the waveform
    sd.play(waveform, samplerate=sample_rate, blocksize=buffer_size)
    pause_processing = True
    threading.Timer(0.5, resume_processing).start()
    sd.wait()


def audio_callback(indata, frames, time, status):
    global pause_processing, stopsendingconfirm

    if pause_processing:
        return  # Skip processing when paused

    fft_result = np.abs(np.fft.fft(indata[:, 0]))

    indices = np.array(indices_before_numpy)

    # Getting only DTMF relevant magnitudes
    Magnitudes = fft_result[indices]

    # Calculate corresponding frequencies
    Frequencies = indices * 10
    Frequencies = Frequencies.reshape(-1, 1)

    # Combine magnitudes with frequencies
    Graph = np.hstack([Magnitudes.reshape(-1, 1), Frequencies])

    # Sort and select the top frequencies
    sorted_indices = np.argsort(Graph[:, 0])
    top = Graph[sorted_indices][
        -4:
    ]  # 2 for control DTMF. 2 for Value DTMF. 1 Extra for slight noise flexibility
    top_frequencies = top[:, 1]

    ca.append(name_of_matched_pair(control_pairs_p, top_frequencies))
    ra.append(name_of_matched_pair(value_pairs_p, top_frequencies))

    if find_triplicate(ca) is not None:
        if find_triplicate(ra) is not None:
            print(find_triplicate(ca), find_triplicate(ra))
            pause_processing = True
            verifyvalues()
            threading.Timer(0.3, resume_processing).start()
            for _ in range(array_length):
                ca.append("G")


sentvalues = []
correct = False


def verifyvalues():
    global correct, sentvalues, stopsending, stopsendingconfirm
    if (
        sentvalues[0] == find_triplicate(ca)
        and sentvalues[1] == find_triplicate(ra)
        and not stopsending
    ):
        correct = True
        print("Recieved matches sent values")
        sentvalues = []
        stopsending = True
    elif (
        sentvalues[0] == find_triplicate(ca)
        and sentvalues[1] == find_triplicate(ra)
        and stopsending
    ):
        print("Stopped sending confirm")
        stopsendingconfirm = True
        sentvalues = []
    else:
        print("Recieved doesnt match sent values")
        correct = False


def converttotone(number):
    hex_digits = "0123456789ABCDEF"  # Hexadecimal digits

    # Find the quotient and remainder when divided by 16
    quotient = number // 16
    remainder = number % 16

    # Convert both the quotient and the remainder to hexadecimal using the hex_digits string
    hex_number = hex_digits[quotient] + hex_digits[remainder]

    return play_tones(hex_digits[quotient], hex_digits[remainder])


def repeated_send(number):
    global stopsending, sentvalues
    if stopsending:
        return  # Stop sending if the flag is set to True

    converttotone(number)  # Call your function

    # Schedule this function to be called again after 5 seconds
    threading.Timer(1, repeated_send, [number]).start()


"""
THIS IS THE ROBOT MOVEMENT AND POSITION PART
"""
TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]

# Time for movement
last_velocity_time = time.time()

current_angle = 0.0
Senddistance = False


def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    theta_degrees = math.degrees(theta)
    return int(r), int(theta_degrees)


def set_resolution(capture, width, height):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detecting one hand to reduce complexity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def update_position_and_angle(linear_velocity, angular_velocity, elapsed_time):
    global cor_x, cor_y, current_angle

    if angular_velocity == 0.4:
        angular_velocity = 0.3
    elif angular_velocity == -0.4:
        angular_velocity = -0.3

    # Update the current angle
    angular_displacement = angular_velocity * elapsed_time
    current_angle += angular_displacement
    current_angle = current_angle % (2 * math.pi)  # Normalize the angle

    # Calculate movement direction based on the current angle
    delta_x = math.cos(current_angle)
    delta_y = math.sin(current_angle)

    # Update position based on movement direction and linear velocity
    cor_x += delta_x * linear_velocity * elapsed_time
    cor_y += delta_y * linear_velocity * elapsed_time


# Function to calculate distance between two points in 2D space
def calculate_distance_of_hand(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# Data points for a polynomial regression
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

cap = cv2.VideoCapture(0)
set_resolution(cap, 1920, 1080)

rclpy.init()
qos = QoSProfile(depth=10)
node = rclpy.create_node("rb3_node")
pub = node.create_publisher(Twist, "cmd_vel", qos)

target_angular_velocity = 0.0
target_linear_velocity = 0.0


def robot_movement():
    global last_velocity_time, isopen, distanceflag, Senddistance, stopsending, stopsendingconfirm, angleflag, setdistance
    while True:
        isopen, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        position = "middle"
        distanceCM = 0
        loop_start_time = time.time()
        # Check if any hand landmarks are detected if True then the robot moves if false then the robot stops and if Senddistance is true then send the coordinates in polar form.
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lmList = [
                    (id, int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                    for id, lm in enumerate(handLms.landmark)
                ]
                x, y = lmList[5][1], lmList[5][2]
                x2, y2 = lmList[17][1], lmList[17][2]
                dis = calculate_distance_of_hand(x, y, x2, y2)
                h, w, c = frame.shape
                A, B, C = coff
                distanceCM = A * dis**2 + B * dis + C
                wrist_landmark = hand_landmarks.landmark[0]
                cx = int(wrist_landmark.x * w)
                if cx < w * 0.25:
                    position = "left"
                elif cx > w * 0.75:
                    position = "right"
                target_angular_velocity = 0.0
                target_linear_velocity = 0.0
            if distanceCM > 50 and position == "middle":
                target_linear_velocity = 0.1
            elif position == "left" and distanceCM > 50:
                target_angular_velocity = -0.4
                target_linear_velocity = 0.15
            elif distanceCM < 20:
                target_linear_velocity = -0.20
            elif position == "right" and distanceCM > 50:
                target_angular_velocity = 0.4
                target_linear_velocity = 0.15
            elif position == "right":
                target_angular_velocity = 0.39
            elif position == "left":
                target_angular_velocity = -0.39
            else:
                target_angular_velocity = 0.0
                target_linear_velocity = 0.0
                Senddistance = True
        else:
            target_angular_velocity = 0.0
            target_linear_velocity = 0.0
            twist = Twist()
            twist.linear.x = target_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = target_angular_velocity
            pub.publish(twist)
            if Senddistance and cor_x > 1:
                r, theta = cartesian_to_polar(int(cor_x), int(cor_y))
                distanceflag = True
                repeated_send(r)
                repeated_confirm()
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=sample_rate,
                    channels=1,
                    blocksize=frame_size,
                ):
                    print("Audio stream started. Press Ctrl+C to stop...")
                    sd.sleep(8000)
                distanceflag = False
                stopsending = False
                stopsendingconfirm = False
                angleflag = True
                repeated_send(theta)
                repeated_confirm()
                with sd.InputStream(
                    callback=audio_callback,
                    samplerate=sample_rate,
                    channels=1,
                    blocksize=frame_size,
                ):
                    print("Audio stream started. Press Ctrl+C to stop...")
                    sd.sleep(8000)
                angleflag = False
                stopsending = False
                stopsendingconfirm = False
                Senddistance = False

        time_since_last_update = loop_start_time - last_velocity_time
        update_position_and_angle(
            target_linear_velocity, target_angular_velocity, time_since_last_update
        )
        last_velocity_time = loop_start_time
        angle_in_degrees = math.degrees(current_angle)
        print(
            f"Justeret vinkel i grader: {angle_in_degrees:.2f}, X-Value: {cor_x}, Y-Value: {cor_y}"
        )
	# Twist functions for robot movement
        twist = Twist()
        twist.linear.x = target_linear_velocity
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = target_angular_velocity
        pub.publish(twist)


robot_movement()

