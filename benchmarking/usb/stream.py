from usb.packet import UsbPacket

class StreamContext:
    def __init__(self):
        self.interrupts = []
        self.BEGIN = False
        self.ENDPOINTS = {
                "BULK"      : 1,
                "SPECIAL"   : 2,
                "INTERRUPT" : 3,
        }

    def stream_valid(self, p:UsbPacket):
        return (p.valid_communication == True)

    def stream_started(self, p:UsbPacket):
        if (p.is_host_src() and
            p.endpoint == self.ENDPOINTS["INTERRUPT"]):
            return True
        return False

