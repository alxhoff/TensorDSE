from usb.timer import UsbTimer
from typing import Dict

def analyze_timestamps(t:UsbTimer) -> Dict:
    d = {
            "send" : 0.0,
            "recv" : 0.0,
            "inf"  : 0.0,
            "usb" : {}
    }

    d["usb"]["host_submission"]    = float(t.ts_end_submission - t.ts_begin_submission)
    d["usb"]["host_communication"] = float(t.ts_end_host_send_request - t.ts_begin_host_send_request)
    d["usb"]["tpu_communication"]  = float(t.ts_begin_return - t.ts_end_submission)
    d["usb"]["tpu_return"]         = float(t.ts_end_return - t.ts_end_submission)
    d["usb"]["inference"]          = float(t.ts_begin_return - t.ts_end_submission)
    d["usb"]["total"]              = float(t.ts_end_return - t.ts_begin_host_send_request)

    for k in d["usb"]:
        if d["usb"][k] < 0:
            return {}

    d["send"] = d["usb"]["host_communication"] + d["usb"]["host_submission"]
    d["recv"] = d["usb"]["tpu_return"]
    d["inf"]  = d["usb"]["inference"]

    return d

