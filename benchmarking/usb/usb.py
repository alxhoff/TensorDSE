class UsbPacket:
    """Class containing all necessary methods to decode/retreive human
    understandable information regarding incoming usb packets.

    Not only does these methods decode usb info but also sometimes exposes them
    as easy to use booleans that are useful in conditional statements.
    """

    def __init__(self, raw_packet, id):
        self.valid_data     = False
        self.present_data   = False
        self.id             = id

        self.timestamp(raw_packet)
        self.find_scr_dest(raw_packet)
        self.find_direction(raw_packet)
        self.find_transfer_type(raw_packet)
        self.find_urb_type(raw_packet.usb.urb_type)

    def timestamp(self, packet):
        self.ts = float(packet.frame_info.time_relative)

    def find_direction(self, packet):
        """
        Method that stores the overloaded packet's flag denoting direction
        of usb transfer.
        """
        self.direction = packet.usb.endpoint_address_direction

    def find_scr_dest(self, packet):
        """Stores source and destination values of overloaded packet."""
        self.src = packet.usb.src
        self.dest = packet.usb.dst

    def find_transfer_type(self, packet):
        """Finds the urb transfer type of the overloaded packet."""
        hexa_transfer_type = packet.usb.transfer_type
        default = "OTHER"
        transfer_dict = {
            "0x00000001":   "INTERRUPT",
            "0x00000002":   "CONTROL",
            "0x00000003":   self.find_bulk_type(),
        }

        self.transfer_type = transfer_dict.get(hexa_transfer_type, default)

    def find_urb_type(self, urb_type):
        """Finds the urb type of the overloaded packet."""
        default = None
        transfer_dict = {
            "'S'":   "SUBMIT",
            "'C'":   "COMPLETE",
            "":   None
        }

        self.urb_type = transfer_dict.get(urb_type, default)

    def find_bulk_type(self):
        """Finds the packet's usb bulk variation with use of its direction var."""
        if self.direction == '1':
            return "BULK IN"
        elif self.direction == '0':
            return "BULK OUT"
        else:
            raise ValueError(
                "Wrong attribute type in packet.endpoint_address_direction.")

    def verify_src(self, string):
        """Verifies if the packet's source is one that is valid for our analysis."""
        valid = f"{self.id}.3"
        if string == "begin":
            return True if ("host" == self.src and valid == self.dest) else False

        elif string == "end":
            return True if (valid == self.src and "host" == self.dest) else False

        else:  # both
            return True if (((self.id in self.src) and ("host" == self.dest))
                            or (("host" == self.src) and (self.id in self.dest))) else False

    def verify_data_presence(self, packet):
        """Finds if the overloaded packet contains actual DATA being sent."""

        tmp = packet.usb.data_flag
        if tmp == '>' or tmp == '<':
            self.present_data = False
            return False
        elif 'not present' in tmp:
            self.present_data = False
            return False
        elif tmp == 'present (0)':
            self.present_data = True
            return True
        else:
            raise ValueError("Unknown data presence variable.")

    def verify_data_validity(self, packet):
        """Finds if the overloaded packet contains VALID data being sent."""
        self.urb_size = float(packet.usb.urb_len)
        self.data_size = float(packet.usb.data_len)

        if self.src != "host":
            if self.data_size > 0 and self.data_size >= 16:
                self.valid_data = True
                return True
            else:
                self.valid_data = False
                return False
        else:
            if self.data_size > 0:
                self.valid_data = True
                return True
            else:
                self.valid_data = False
                return False
