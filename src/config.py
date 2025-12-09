import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Cheating Scenario Dataset in Online Exam', 'Cheating Scenario Dataset in Online Exam')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Model
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
CONFIDENCE_THRESHOLD = 0.5
# Device selection
# "auto" will pick GPU if available else CPU. Use "0" to force first GPU, "cpu" to force CPU.
DEVICE = "auto"

# 13 Classes - 4 Categories
CLASS_NAMES = [
    'head_down',        # 0 - Head Pose
    'head_up',          # 1 - Head Pose
    'head_left',        # 2 - Head Pose
    'head_right',       # 3 - Head Pose
    'head_center',      # 4 - Head Pose
    'eye_center',       # 5 - Eye Gaze
    'eye_left',         # 6 - Eye Gaze
    'eye_right',        # 7 - Eye Gaze
    'eye_up',           # 8 - Eye Gaze
    'eye_down',         # 9 - Eye Gaze
    'eye_closed',       # 10 - Eye Gaze
    'lip_closed',       # 11 - Lip Movement
    'lip_open',         # 12 - Lip Movement
]

# Cheating detection rules
CHEATING_INDICATORS = {
    'head': [0, 1, 2, 3],      # head_down, head_up, head_left, head_right
    'eye': [6, 7, 8, 9],       # eye_left, eye_right, eye_up, eye_down
    'lip': [12],               # lip_open (speaking)
}

NORMAL_INDICATORS = {
    'head': [4],               # head_center
    'eye': [5],                # eye_center
    'lip': [11],               # lip_closed
}

# Training
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42
