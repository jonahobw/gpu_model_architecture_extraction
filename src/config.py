"""Configuration of profile collection and model construction.

This module contains configuration settings for GPU profile collection and model construction.
It defines valid models, system signals to monitor, input parameters, and email settings.

Dependencies:
    - email_sender: For email notification functionality
    - get_model: For model name to family mapping

Example Usage:
    ```python
    from config import MODELS, SYSTEM_SIGNALS
    
    # Get list of models to profile
    models_to_profile = MODELS
    
    # Get system signals to monitor
    signals = SYSTEM_SIGNALS
    ```
"""

from typing import List, Dict, Any
from email_sender import EmailSender
from get_model import name_to_family

# List of all valid model names from the model family mapping
VALID_MODELS: List[str] = list(name_to_family.keys())

# Models for which to collect profiles. Can be a subset of VALID_MODELS
MODELS: List[
    str
] = VALID_MODELS  # MODELS = ["googlenet", "mobilenetv3", "resnet", "vgg"]

# System signals to monitor during profiling
SYSTEM_SIGNALS: List[str] = [
    "sm_clock_(mhz)",  # Streaming Multiprocessor clock speed
    "memory_clock_(mhz)",  # GPU memory clock speed
    "temperature_(c)",  # GPU temperature
    "power_(mw)",  # Power consumption
    "fan_(%)",  # Fan speed percentage
]

# Input image parameters
CHANNELS: int = 3  # Number of color channels (RGB)
INPUT_SIZE: int = 224  # Input image size (224x224)

# Email configuration for notifications
EMAIL_CONF: Dict[str, Any] = {
    "sender": "kundu.lab.keb310@gmail.com",  # Sender email address
    "pw": "email_pw.txt",  # Path to password file
    "reciever": "jobrienweiss@umass.edu",  # Recipient email address
    "send": True,  # Whether to send emails
}

# Initialize email sender with configuration
EMAIL: EmailSender = EmailSender(**EMAIL_CONF)
