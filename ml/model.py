import joblib 
import json 
import numpy as np 
import pandas as pd 
from pathlib import Path

MODEL_DIR = Path(__file__).parent

CONFIDENCE_THRESHOLD = 0.70

