"""
Missing  Docstring: TODO
"""

from utils.usb.packet import UsbPacket

class StreamContext:
    """
    Missing  Docstring: TODO
    """

    def __init__(self):
        self.interrupts = []
        self.timeout = False
        self.max_timeout = False
        self.current_phase = 'WAITING_IN'

        self.data_traffic = False
        self.host_data = []
        self.tpu_data = []

        self.begin = False
        self.end = False

        self.endpoints = {
                "BULK"      : 1,
                "SPECIAL"   : 2,
                "INTERRUPT" : 3,
        }

        self.timestamps = {
            "begin"     : 0.0,
            "host_data" : {
                "first" : 0.0,
                "last"  : 0.0
            },
            "tpu_data" : {
                "first" : 0.0,
                "last"  : 0.0
            },
            "end"     : 0.0,
        }

    def __start_condition(self, p:UsbPacket) -> bool:
        if (p.is_host_src() and
            p.endpoint == self.endpoints["INTERRUPT"]):
            return True
        return False

    def __end_condition(self, p:UsbPacket) -> bool:
        if (p.is_tpu_src() and
            self.begin and
            p.endpoint == self.endpoints["INTERRUPT"]):
            return True
        return False

    def stream_valid(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        return p.valid_communication is True

    def stream_started(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        if not self.begin:
            if self.__start_condition(p):
                self.begin = True
                self.timestamps["begin"] = p.ts
                return True
            return False
        return True

    def has_data_trafficked(self) -> bool:
        """
        Missing  Docstring: TODO
        """
        if not self.data_traffic and len(self.host_data) > 0:
            self.data_traffic = True
            return True
        return False

    def stream_ended(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        if not self.end:
            if self.__end_condition(p):
                self.end = True
                self.timestamps["end"] = p.ts
                return True
            return False
        return True

    def contains_host_data(self, p:UsbPacket) -> bool:
        """docstring for contains_host_data"""
        if (p.is_host_src() and
            p.transfer_type == "BULK OUT" and
            p.is_data_valid()):
            return True
        return False

    def timestamp_host_data(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        if len(self.tpu_data) == 0:
            self.host_data.append(p.ts)

    def contains_tpu_data(self, p:UsbPacket) -> bool:
        """docstring for contains_host_data"""
        if (p.is_tpu_src() and
            p.transfer_type == "BULK IN" and
            p.is_data_valid()):
            return True
        return False

    def timestamp_tpu_data(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        self.tpu_data.append(p.ts)

    def set_timestamps(self):
        """
        Missing  Docstring: TODO
        """
        self.timestamps["host_data"]["first"] = self.host_data[0]
        self.timestamps["host_data"]["last"]  = self.host_data[-1]
        self.timestamps["tpu_data"]["first"]  = self.tpu_data[0]
        self.timestamps["tpu_data"]["last"]   = self.tpu_data[-1]

        return self.timestamps

    def timed_out(self):
        """
        Missing  Docstring: TODO
        """
        self.timeout = True

    def maxed_out(self):
        """
        Missing  Docstring: TODO
        """
        self.max_timeout = True

    def is_successful(self):
        """
        Missing  Docstring: TODO
        """
        if len(self.host_data) > 0 and len(self.tpu_data) > 0:
            return True
        elif self.timed_out or self.max_timeout:
            return False
        else:
            return False

    def conclude(self):
        """
        Missing  Docstring: TODO
        """
        if len(self.host_data) > 0 and len(self.tpu_data) > 0:
            return self.set_timestamps()

        elif self.timed_out():
            return {
                "error" : {
                    "reason" : "timed out without correct traffic analysis",
                },
            }

        elif self.max_timeout:
            return {
                "error" : {
                    "name" : "maxed_out",
                    "reason" : "timed out with no traffic found"
                },
            }

        else:
            return {
                "error" : {
                    "name" : "emergency",
                    "reason" : "unknown"
                },
            }

    def set_phase(self, p:UsbPacket):
        """
        Missing  Docstring: TODO
        """
        if self.current_phase == 'WAITING_IN':
            if ('BULK' in p.transfer_type) or ('INTERRUPT' in p.transfer_type):
                self.current_phase = 'TRANSFER'
        elif self.current_phase == 'TRANSFER':
            if 'CONTROL' in p.transfer_type:
                self.current_phase = 'WAITING_OUT'

    def is_inference_ended(self):
        """
        Missing  Docstring: TODO
        """
        return self.current_phase == 'WAITING_OUT'
