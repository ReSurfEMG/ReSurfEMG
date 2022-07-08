from threading import Lock


_singleton = None
_lock = Lock()


def get_singleton():
    with _lock:
        global _singleton

        if _singleton is None:
            _singleton = Variables()

        return _singleton


class Variables:

    def __init__(self):
        self.emg = None
        self.emg_freq = None
        self.ventilator_freq = None

    def set_emg(self, emg):
        self.emg = emg

    def set_emg_freq(self, emg_freq):
        self.emg_freq = emg_freq

    def get_emg_freq(self):
        return self.emg_freq

    def set_ventilator_freq(self, ventilator_freq):
        self.ventilator_freq = ventilator_freq

    def get_ventilator_freq(self):
        return self.ventilator_freq
