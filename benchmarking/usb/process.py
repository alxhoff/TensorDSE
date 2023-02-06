from typing import Dict, Tuple
from json.encoder import py_encode_basestring
from analysis.analysis import Analyzer

from typing import Dict, List

def faulty_timestamps(timestamps:List) -> Tuple[bool, Dict, int]:
    d = {}
    total_errors = 0
    for t in timestamps:
        for k in t:
            if k == "error":
                total_errors += 1
                name = t["error"]["name"]
                if name in t:
                    d[name]["count"] = d[name]["count"] + 1
                else:
                    d[name] = {
                            "reason" : t["error"]["reason"],
                            "count"  : 0
                    }
    if d == {}:
        return False, {}, 0
    return True, d, total_errors

def process_streams(timestamps:List, py_results:List) -> Dict:
    if len(timestamps) == 0:            # cpu, gpu
        return {
                "communication" : {
                    "mean" : 0.0
                },
        }

    faulty, errors, nr_errors = faulty_timestamps(timestamps)   # faulty tpu analysis
    if faulty:
        return {
                "communication" : {
                    "mean" : 0.0
                },
                "errors" : errors,
                "error_runs" : nr_errors,
                "total_runs"  : len(py_results)
        }

    # functional tpu analysis
    total           = [ t["tpu_data"]["last"] - t["host_data"]["first"] for t in timestamps ]
    send            = [ t["host_data"]["last"] - t["host_data"]["first"] for t in timestamps ]
    recv            = [ t["tpu_data"]["last"] - t["tpu_data"]["first"] for t in timestamps ]
    inference       = [ t["tpu_data"]["first"] - t["host_data"]["last"] for t in timestamps ]
    communication   = [ t - i for t,i in zip(total, inference) ]

    t = Analyzer(total)
    s = Analyzer(send)
    r = Analyzer(recv)
    i = Analyzer(inference)
    c = Analyzer(communication)

    return {
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
            },
            "communication" : {
                    "mean"                      : c.mean,
                    "median"                    : c.median,
                    "standard_deviation"        : c.std_deviation,
                    "avg_absolute_deviation"    : c.avg_absolute_deviation,
            },
            "inference" : {
                    "mean"                      : i.mean,
                    "median"                    : i.median,
                    "standard_deviation"        : i.std_deviation,
                    "avg_absolute_deviation"    : i.avg_absolute_deviation,
            },
            "total" : {
                    "mean"                      : t.mean,
                    "median"                    : t.median,
                    "standard_deviation"        : t.std_deviation,
                    "avg_absolute_deviation"    : t.avg_absolute_deviation,
            },
    }
