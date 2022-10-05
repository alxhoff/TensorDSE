class UsbPacket:
    """Class containing all necessary methods to decode/retreive human
    understandable information regarding incoming usb packets.

    Not only does these methods decode usb info but also sometimes exposes them
    as easy to use booleans that are useful in conditional statements.
    """

    def __init__(self, raw_packet, id, addr):
        self.id             = id
        self.addr           = addr
        self.ts             = 0.0

        self.direction      = 0
        self.src            = ""
        self.dest           = ""

        self.data_flag      = ""
        self.data_size      = 0.0
        self.urb_size       = 0.0

        self.transfer_type  = ""
        self.urb_type       = ""

        self.valid_comms    = False
        self.present_data   = False
        self.empty_data     = False
        self.valid_data     = False

        self.timestamp(raw_packet)
        self.store_scr_dest(raw_packet)
        self.store_direction(raw_packet)
        self.store_data_len_info(raw_packet)
        self.find_transfer_type(raw_packet)
        self.find_urb_type(raw_packet)
        self.verify_data_presence()
        self.verify_data_validity()
        self.verify_communication()

    def timestamp(self, packet):
        self.ts = float(packet.frame_info.time_relative)

    def store_direction(self, packet):
        """
        Method that stores the overloaded packet's flag denoting direction
        of usb transfer.
        """
        self.direction = packet.usb.endpoint_address_direction
        self.endpoint  = packet.usb.endpoint_address_number

    def store_scr_dest(self, packet):
        """Stores source and destination values of overloaded packet."""
        self.src = packet.usb.src
        self.dest = packet.usb.dst

    def store_data_len_info(self, packet):
        self.data_flag = packet.usb.data_flag
        self.data_size = float(packet.usb.data_len)
        self.urb_size  = float(packet.usb.urb_len)

    def find_transfer_type(self, packet):
        """Finds the urb transfer type of the overloaded packet."""
        transfer_type = packet.usb.transfer_type
        transfer_dict = {
            ""    :   None,
            "0x01":   "INTERRUPT",
            "0x02":   "CONTROL",
            "0x03":   "BULK IN" if self.direction == '1'
                                else "BULK OUT",
        }

        self.transfer_type = transfer_dict.get(
                transfer_type,
                packet.usb.transfer_type)

    def find_urb_type(self, packet):
        """Finds the urb type of the overloaded packet."""
        urb_type = packet.usb.urb_type
        transfer_dict = {
            ""   :   None,
            "'S'":   "SUBMIT",
            "'C'":   "COMPLETE",
        }

        self.urb_type = transfer_dict.get(urb_type, None)

    def verify_communication(self):
        """Verifies if the packet's source is one that is valid for our analysis."""
        if ((self.addr in self.src) and ("host" == self.dest)):
            self.valid_communication = True
        elif (("host" == self.src) and (self.addr in self.dest)):
            self.valid_communication = True
        else:
            self.valid_communication = False

    def verify_data_presence(self):
        """Finds if the overloaded packet contains actual DATA being sent."""
        if self.data_flag in ('>', '<'):
            self.present_data = False
        elif self.data_flag == 'present (0)':
            self.present_data = True
        elif 'not present' in self.data_flag:
            self.present_data = False
        else:
            raise ValueError("Unknown data presence variable.")

    def verify_data_validity(self):
        """Finds if the overloaded packet contains VALID data being sent."""
        if self.data_size > 0:
            self.valid_data = True
        elif self.present_data:
            self.empty_data = True

    def is_host_src(self):
        return ("host" == self.src)

    def is_host_dest(self):
        return ("host" == self.dest)

    def is_tpu_src(self):
        return (self.addr in self.src)

    def is_tpu_dest(self):
        return (self.addr in self.dest)

    def is_data_valid(self):
        return (self.present_data and self.data_size > 0)

    def is_data_present(self):
        return (self.present_data == True)
