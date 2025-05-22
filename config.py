"""
Configuration of profile collection and model construction.
"""
from email_sender import EmailSender
from get_model import name_to_family

VALID_MODELS = list(name_to_family.keys())

# models for which to collect profiles for, may be a subset of all valid models
MODELS = VALID_MODELS  # MODELS = ["googlenet", "mobilenetv3", "resnet", "vgg"]
SYSTEM_SIGNALS = [
    "sm_clock_(mhz)",
    "memory_clock_(mhz)",
    "temperature_(c)",
    "power_(mw)",
    "fan_(%)",
]
CHANNELS = 3
INPUT_SIZE = 224

EMAIL_CONF = {
    "sender": "kundu.lab.keb310@gmail.com",
    "pw": "email_pw.txt",
    "reciever": "jobrienweiss@umass.edu",
    "send": True,
}
EMAIL = EmailSender(**EMAIL_CONF)
