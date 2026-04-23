# config.py

STAR_NAME = "HD 209458"
MISSION = "TESS"

# Preprocessing
WINDOW_LENGTH = 201
SIGMA_CLIP = 5.0

# BLS Detection
PERIOD_MIN = 0.5
PERIOD_MAX = 15.0
PERIOD_STEPS = 10000
TRANSIT_DURATIONS = [0.05, 0.1, 0.15, 0.2]  # days

# Phase folding
PHASE_WIDTH = 0.02

# Known stellar parameters (HD 209458)
STAR_RADIUS_SOLAR = 1.2       # in solar radii
PLANET_RADIUS_JUP = 1.359     # in Jupiter radii (NASA reference)
STAR_MASS_SOLAR = 1.148       # in solar masses
STAR_TEFF = 6065              # Kelvin

# ML
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_AUGMENTATIONS = 50          # synthetic samples per class

# Paths
DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
MODEL_PATH = "models/xgb_model.pkl"