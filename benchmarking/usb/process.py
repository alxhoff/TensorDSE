from usb.timer import UsbTimer
from analysis.analysis import Analyzer

from typing import Dict, List

def process_timers(arr:List) -> Dict:
    if len(arr) == 0:
        return {
                "send" : 0.0,
                "recv" : 0.0,
        }

    if (not "send" in arr[0] or not "recv" in arr[0]):
        return {
                "send" : 0.0,
                "recv" : 0.0,
        }

    send = []
    recv = []
    for a in arr:
        send.append(a["send"])
        recv.append(a["recv"])

    s = Analyzer(send)
    r = Analyzer(recv)
    d = {
            "send" : {
                    "mean"                      : s.mean,
                    "median"                    : s.median,
                    "standard_deviation"        : s.std_deviation,
                    "avg_absolute_deviation"    : s.avg_absolute_deviation,
            },
            "recv" : {
                    "mean"                      : r.mean,
                    "median"                    : r.median,
                    "standard_deviation"        : r.std_deviation,
                    "avg_absolute_deviation"    : r.avg_absolute_deviation,
            }
    }
    return d

def process_timestamps(t:UsbTimer) -> Dict:
    d = {}
    r = {
            "send" : 0.0,
            "recv" : 0.0,
    }

    d["host_communication"] = float(t.ts_end_host_send_request - t.ts_begin_host_send_request)
    d["host_data"]          = float(t.ts_end_submission - t.ts_begin_submission)
    d["tpu_communication"]  = float(t.ts_begin_return - t.ts_end_submission)
    d["tpu_data"]           = float(t.ts_end_return - t.ts_end_submission)
    d["inference"]          = float(t.ts_begin_return - t.ts_end_submission)
    d["total"]              = float(t.ts_end_return - t.ts_begin_host_send_request)

    for k in d:
        if d[k] < 0:
            return {}

    r["send"] = d["host_communication"] + d["usb"]["host_data"]
    r["recv"] = d["tpu_return"]

    return r

