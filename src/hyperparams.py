import math

def get_frames(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T


vocab='РЕ абвгдеёжзийклмнопрстуфхцчшщъыьэюя.?'
# signal processing
sample_rate = 22050 # Sampling rate.
silence_threshold = 2
fmin = 125
fmax = 7600
n_fft = 2048 # fft points (samples)
hop_size = 256
frame_shift_ms = None
win_length = 1024
win_length_ms = -1
window = 'hann'
highpass_cutoff = 70
n_mels = 80 # Number of Mel banks to generate
sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
n_iter = 50 # Number of inversion iterations
preemphasis = .97 # or None
min_db = -100
ref_db = 20
allow_clipping_in_normalization = True


# MODEL
dropout= 1 - 0.95
r = 4 # Reduction factor
# num_filters=5
kernel_size=5
downsample_step=4
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
data_path = '/data/ru_audiobook_single_speaker/'
transcript_path = 'transcript.txt'
max_duration = 10 #seconds
max_timesteps = 100 # max characters throughout sample
max_frames = int(get_frames(max_duration, sample_rate, hop_size, r))

# TRAINING
epochs=50
batch_size=16
lr=0.001
max_grad_norm = 100.
max_grad_val = 5.
num_iterations = 500000
