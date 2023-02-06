from usb.packet import UsbPacket

class StreamContext:
    def __init__(self):
        self.interrupts = []
        self.timeout = False
        self.max_timeout = False

        self.BEGIN = False
        self.END = False

        self.ENDPOINTS = {
                "BULK"      : 1,
                "SPECIAL"   : 2,
                "INTERRUPT" : 3,
        }

        self.data_traffic = False
        self.host_data = []
        self.tpu_data = []

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
            p.endpoint == self.ENDPOINTS["INTERRUPT"]):
            return True
        return False

    def __end_condition(self, p:UsbPacket) -> bool:
        if (p.is_tpu_src() and
            self.BEGIN and
            p.endpoint == self.ENDPOINTS["INTERRUPT"]):
            return True
        return False

    def stream_valid(self, p:UsbPacket):
        return (p.valid_communication == True)

    def stream_started(self, p:UsbPacket):
        if not self.BEGIN:
            if (self.__start_condition(p)):
                self.BEGIN = True
                self.timestamps["begin"] = p.ts
                return True
            return False
        return True

    def has_data_trafficked(self) -> bool:
        if not self.data_traffic and len(self.host_data) > 0:
            self.data_traffic = True
            return True
        return False

    def stream_ended(self, p:UsbPacket):
        if not self.END:
            if (self.__end_condition(p)):
                self.END = True
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
            self.tpu_data.append(p.ts)

    def set_timestamps(self):
        self.timestamps["host_data"]["first"] = self.host_data[0]
        self.timestamps["host_data"]["last"]  = self.host_data[-1]
        self.timestamps["tpu_data"]["first"]  = self.tpu_data[0]
        self.timestamps["tpu_data"]["last"]   = self.tpu_data[-1]

        return self.timestamps

    def timed_out(self):
        self.timeout = True

    def maxed_out(self):
        self.max_timeout = True

    def is_successful(self):
        if len(self.host_data) > 0 and len(self.tpu_data) > 0:
            return True
        elif self.timed_out or self.max_timeout:
            return False
        else:
            return False

    def conclude(self):
        if len(self.host_data) > 0 and len(self.tpu_data) > 0:
            return self.set_timestamps()

        elif self.timed_out :
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
