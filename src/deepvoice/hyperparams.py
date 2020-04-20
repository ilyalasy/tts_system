import math

def get_frames(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T


vocab='РЕ абвгдеёжзийклмнопрстуфхцчшщъыьэюя-.?'
# signal processing
sr = 22050 # Sampling rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
n_mels = 80 # Number of Mel banks to generate
sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
n_iter = 50 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20

# MODEL
dropout=0.95
r = 4 # Reduction factor
num_filters=5
kernel_size=3
# ENCODER
embed_size=256
num_channels=64
encoder_layers=7
# DECODER
decoder_layers = 4
attention_size = 256 # == a
# CONVERTER 
converter_layers = 5
converter_channels = 256 # == v

# DATA
max_duration = 10 #seconds
max_timesteps = 180 # max characters in sample
max_frames = int(get_frames(max_duration, sr, hop_length, r))

