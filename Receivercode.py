import sounddevice as sd
import numpy as np
import threading
import math

def degrees_to_radians(degrees):
    return degrees * math.pi / 180

coordinates = []

def convert_to_coordinates(anglearray, distancearray):
    global coordinates
    for angle, distance in zip(anglearray, distancearray):
        radian_angle = degrees_to_radians(angle)
        x = distance * math.cos(radian_angle)
        y = distance * math.sin(radian_angle)
        coordinates.append((x, y))
    return coordinates

anglearray = []
distancearray= []
coordinatesarray = []

angle = 0
distance = 0
pause_processing = False
sample_rate = 5120
frame_size = 512
array_length = 5
to_check = 3
newvalueflag = False
fixpreviouspointerflag = False

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

ra=RollingArray()
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

ca=ControlArray()
for _ in range(array_length):
    ca.append("G")


def find_triplicate(arr):
    if len(arr) != array_length:
        return "Invalid array size"

    frequency_count = {}
    
    for ele in arr:
        if ele != "G":
            frequency_count[ele] = frequency_count.get(ele, 0) + 1

    # Filter the dictionary
    candidates = {key: value for key, value in frequency_count.items() if value >= to_check}
    
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
        return matches[0]  # Return the name of the matched pair if only one match is found
    return "G"  # Return "G" if no match or multiple matches are found


value_pairs_r = {
    "0": (700, 1210),    "1": (700, 1340),    "2": (700, 1480),    "3": (700, 1630),
    "4": (770, 1210),    "5": (770, 1340),    "6": (770, 1480),    "7": (770, 1630),
    "8": (850, 1210),    "9": (850, 1340),    "A": (850, 1480),    "B": (850, 1630),
    "C": (940, 1210),    "D": (940, 1340),    "E": (940, 1480),    "F": (940, 1630)
    }

control_pairs_r = {
    "0": (400, 1700),    "1": (400, 1770),    "2": (400, 1840),    "3": (400, 1910),
    "4": (470, 1700),    "5": (470, 1770),    "6": (470, 1840),    "7": (470, 1910),
    "8": (550, 1700),    "9": (550, 1770),    "A": (550, 1840),    "B": (550, 1910),
    "C": (620, 1700),    "D": (620, 1770),    "E": (620, 1840),    "F": (620, 1910)
    }

value_pairs_p = {
    "A": (700, 1210),    "B": (700, 1340),    "8": (700, 1480),    "9": (700, 1630),
    "E": (770, 1210),    "F": (770, 1340),    "C": (770, 1480),    "D": (770, 1630),
    "2": (850, 1210),    "3": (850, 1340),    "0": (850, 1480),    "1": (850, 1630),
    "6": (940, 1210),    "7": (940, 1340),    "4": (940, 1480),    "5": (940, 1630)
    }

control_pairs_p = {
    "A": (400, 1700),    "B": (400, 1770),    "8": (400, 1840),    "9": (400, 1910),
    "E": (470, 1700),    "F": (470, 1770),    "C": (470, 1840),    "D": (470, 1910),
    "2": (550, 1700),    "3": (550, 1770),    "0": (550, 1840),    "1": (550, 1910),
    "6": (620, 1700),    "7": (620, 1770),    "4": (620, 1840),    "5": (620, 1910)
    }

indices_before_numpy = [40, 47, 55, 62, 70, 77, 85, 94, 121, 134, 148, 163, 170, 177, 184, 191] # extend for purposeful noise inclusion

def play_tones(key1, key2, duration=0.5, buffer_size=2048, sample_rate=40960):
    # Retrieve frequencies from the dictionaries
    freq1 = control_pairs_r.get(key1)
    freq2 = value_pairs_r.get(key2)

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
    sd.wait()
    

def play_response(key1, key2, duration=0.5, buffer_size=2048, sample_rate=40960):
    global indices_before_numpy
    # Retrieve frequencies from the dictionaries
    freq1 = control_pairs_p.get(key1)
    freq2 = value_pairs_p.get(key2)
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
    sd.wait()
    sd.sleep(50)
    indices_before_numpy = [40, 47, 55, 62, 70, 77, 85, 94, 121, 134, 148, 163, 170, 177, 184, 191] # reset array

def audio_callback(indata, frames, time, status):
    # Listens through default microphone
    global pause_processing
    global distance
    global angle
    global result
    global newvalueflag
    global fixpreviouspointerflag
    global coordinates
    global coordinatesarray
    

    if pause_processing:
        return  # Skip processing when paused

    fft_result = np.abs(np.fft.fft(indata[:, 0]))

    indices = np.array(indices_before_numpy)

    Magnitudes = fft_result[indices]

    Frequencies = indices * 10
    Frequencies = Frequencies.reshape(-1, 1)
    # Combine magnitudes with frequencies
    Graph = np.hstack([Magnitudes.reshape(-1, 1), Frequencies])
    # Sort and select the top 4 frequencies
    sorted_indices = np.argsort(Graph[:, 0])
    top = Graph[sorted_indices][-4:] 
    top_frequencies = top[:, 1]

    ca.append(name_of_matched_pair(control_pairs_r , top_frequencies))
    ra.append(name_of_matched_pair(value_pairs_r , top_frequencies))
    
    if find_triplicate(ca) is not None and find_triplicate(ca) != "F":
        if find_triplicate(ra) is not None:
            print("Hex:", find_triplicate(ca),find_triplicate(ra))
            result=combine_and_convert(find_triplicate(ca),find_triplicate(ra))
            newvalueflag = True
            print(result)
            play_response(find_triplicate(ca),find_triplicate(ra))
            pause_processing = True
            threading.Timer(0.2, resume_processing).start()
            for _ in range(array_length):
                ca.append("G")

    if find_triplicate(ca) == "F":

        if find_triplicate(ra) == "0":
            print("Hex:", find_triplicate(ca),find_triplicate(ra))
            print("stay on pointer:",result)
            if newvalueflag==True:
                print("setting distance, staying on pointer")
                newvalueflag=False
            else:
                print("undo last store. Set distance, stay on pointer")
            play_response(find_triplicate(ca),find_triplicate(ra))
            pause_processing = True
            threading.Timer(0.2, resume_processing).start()
            for _ in range(array_length):
                ca.append("G")

        elif find_triplicate(ra) == "5":
            print("Hex:", find_triplicate(ca),find_triplicate(ra))
            distance = result
            print("Distance:", distance)
            distancearray.append(distance)
            if newvalueflag==True:
                print("set distance")
                newvalueflag=False
            else:
                print("undo last store. Set distance, angle 0")
            newvalueflag=False
            fixpreviouspointerflag=True
            play_response(find_triplicate(ca),find_triplicate(ra))
            pause_processing = True
            threading.Timer(0.2, resume_processing).start()
            for _ in range(array_length):
                ca.append("G")

        elif find_triplicate(ra) == "A":
            print("Hex:", find_triplicate(ca),find_triplicate(ra))
            angle = result
            print("Angle:", angle)
            anglearray.append(angle)
            if newvalueflag==True:
                print("setting angle")
                newvalueflag=False
            else:
                print("undo last store. Distance 0, setting angle")
            newvalueflag=False
            fixpreviouspointerflag=True
            play_response(find_triplicate(ca),find_triplicate(ra))
            pause_processing = True
            threading.Timer(0.2, resume_processing).start()
            for _ in range(array_length):
                ca.append("G")

    print("array of recieved angles:", anglearray)
    print("array of recieved distances:", distancearray)

    # Debugging:
    #print(top_frequencies)

def combine_and_convert(hex1, hex2):
    # Concatenate the hexadecimal strings
    combined_hex = hex1 + hex2

    # Convert the concatenated hexadecimal string to an integer
    return int(combined_hex, 16)

def resume_processing():
    global pause_processing
    pause_processing = False
    print("Resuming processing...")

with sd.InputStream(callback=audio_callback, samplerate=sample_rate, channels=1, blocksize=frame_size):
    print("Press Ctrl+C to stop...")
    sd.sleep(1000000)