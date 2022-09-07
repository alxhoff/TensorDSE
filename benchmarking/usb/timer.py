class UsbTimer:
    """Class containing all necessary time stamps regarding usb traffic and
    their methods.

    As usb traffic is read during the edge_tpu deployment. These methods(functions)
    will either simply save the overloaded timestamps onto the class instance or
    perform minor operations to deduce other relevant information.

    Attributes
    -------
    """

    def __init__(self):
        # Absolute Begin
        self.ts_absolute_begin = 0

        # Begin host sent request before actual data was sent to device(TPU).
        self.ts_begin_host_send_request = 0

        # Last host sent request before actual data was sent to device(TPU).
        self.ts_end_host_send_request = 0

        # Beginning of host sent data to device(TPU).
        self.ts_begin_submission = 0

        # End of host sent data to device(TPU).
        self.ts_end_submission = 0

        # First TPU sent request before actual data was sent to the CPU.
        self.ts_begin_tpu_send_request = 0

        # Last TPU sent request before actual data was sent to the CPU.
        self.ts_end_tpu_send_request = 0

        # End of device(TPU) sent data to host.
        self.ts_begin_return = 0

        # End of device(TPU) sent data to host.
        self.ts_end_return = 0

        # End of Receiving Data
        self.ts_absolute_end = 0

        self.interrupt_begin_src = ""
        self.interrupt_begin_dst = ""
        self.interrupt_end_src = ""
        self.interrupt_end_dst = ""

    def stamp_beginning(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_begin.

        Parameters
        ----------
        packet : object
        This object has as attributes all necessary data regarding an incoming
        usb packet. This will be the same attribute for all stamp-like methods.
        """
        self.ts_absolute_begin = float(packet.frame_info.time_relative)
        self.interrupt_begin_src = packet.usb.src
        self.interrupt_begin_dst = packet.usb.dst

    def stamp_ending(self, packet):
        """Saves the overloaded packet's timestamp onto ts_absolute_end."""
        self.ts_absolute_end = float(packet.frame_info.time_relative)
        self.interrupt_end_src = packet.usb.src
        self.interrupt_end_dst = packet.usb.dst

    def stamp_begin_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_host_send_request."""
        self.ts_begin_host_send_request = float(packet.frame_info.time_relative)

    def stamp_end_host_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_host_send_request."""
        self.ts_end_host_send_request = float(packet.frame_info.time_relative)

    def stamp_beginning_submission(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_submission."""
        self.ts_begin_submission = float(packet.frame_info.time_relative)

    def stamp_beginning_return(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_return."""
        self.ts_begin_return = float(packet.frame_info.time_relative)

    def stamp_begin_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_begin_tpu_send_request."""
        self.ts_begin_tpu_send_request = float(packet.frame_info.time_relative)

    def stamp_end_tpu_send_request(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_tpu_send_request."""
        self.ts_end_tpu_send_request = float(packet.frame_info.time_relative)

    def stamp_src_host(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_submission."""
        self.ts_end_submission = float(packet.frame_info.time_relative)

    def stamp_src_device(self, packet):
        """Saves the overloaded packet's timestamp onto ts_end_return."""
        self.ts_end_return = float(packet.frame_info.time_relative)


