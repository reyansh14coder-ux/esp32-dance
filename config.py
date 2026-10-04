STREAM_URL = "http://10.45.244.47:81/stream"
CAPTURE_URL = "http://10.45.244.47/capture"
FRAME_SIZE = (640, 480)

NUM_KEYPOINTS = 17
SEQUENCE_LENGTH = 60
STRIDE = 15

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DANCE_CLASSES = [
    "bharatanatyam",
    "kathak",
    "odissi",
    "kuchipudi",
    "kathakali",
    "bhangra",
    "garba",
    "mohiniyattam",
    "none",
]

DATASET_DIR = "dataset"
MODEL_PATH = "dance_lstm.pt"
LABEL_MAP_PATH = "label_map.json"

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

ZERO_SHOT_CLASSES = [
    "bharatanatyam", "kathak", "odissi", "kuchipudi", "kathakali",
    "mohiniyattam", "manipuri", "sattriya", "bhangra", "garba",
    "dandiya", "lavani", "chhau", "bollywood", "none",
]

NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.2-11b-vision-instruct"
CLASSIFY_INTERVAL = 2.0
GRID_FRAMES = 16
GRID_SIZE = 1024
MIN_MOVEMENT = 0.6
MIN_VISIBLE = 8
