import json

class Settings(object):
    def __init__(self):
        data = json.load(open('settings.json'))
        self.input_source = data["input_source"]
        self.grid_settings = data["grid_settings"]
        self.plot_colors = data["plot_colors"]
        self.threshold = data["threshold"]
