def capture_stream() -> None:
    """
    """
    import pyshark

    addr = 13

    FILTER = (
    f"usb.transfer_type==URB_BULK || usb.transfer_type==URB_INTERRUPT && usb.device_address=={addr}"
    )

    capture = pyshark.LiveCapture(interface='usbmon0', display_filter=FILTER)

    for raw_packet in capture.sniff_continuously():
        p = raw_packet
        continue

capture_stream()
