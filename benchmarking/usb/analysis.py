from usb.timer import UsbTimer
from typing import Dict

def analyze_timestamps(t:UsbTimer) -> Dict:
    d = {
            "send" : 0.0,
            "recv" : 0.0,
            "inf"  : 0.0,
            "usb" : {}
    }

    d["host_submission"]    = float(t.ts_end_submission - t.ts_begin_submission)
    d["host_communication"] = float(t.ts_end_host_send_request - t.ts_begin_host_send_request)
    d["tpu_communication"]  = float(t.ts_begin_return - t.ts_end_submission)
    d["tpu_return"]         = float(t.ts_end_return - t.ts_end_submission)
    d["inference"]          = float(t.ts_begin_return - t.ts_end_submission)
    d["total"]              = float(t.ts_end_return - t.ts_begin_host_send_request)

    for k in d:
        if d[k] < 0:
            return {}

    d["send"] = d["host_communication"] + d["usb"]["host_submission"]
    d["recv"] = d["tpu_return"]

    return d

