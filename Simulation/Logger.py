import numpy as np
import json

class SignalLogger:
    def __init__(self):
        self.data = {}

    def log(self, name, value):
        v = np.asarray(value)
        stats = {
            "shape": tuple(int(s) for s in v.shape),
            "dtype": str(v.dtype),
            "is_complex": np.iscomplexobj(v),
        }

        if stats["is_complex"]:
            # Real part stats
            stats["real"] = {
                "min": float(np.min(v.real)),
                "max": float(np.max(v.real)),
                "mean": float(np.mean(v.real)),
                "std": float(np.std(v.real)),
            }
            # Imag part stats
            stats["imag"] = {
                "min": float(np.min(v.imag)),
                "max": float(np.max(v.imag)),
                "mean": float(np.mean(v.imag)),
                "std": float(np.std(v.imag)),
            }
            # Magnitude stats
            mag = np.abs(v)
            stats["magnitude"] = {
                "min": float(np.min(mag)),
                "max": float(np.max(mag)),
                "mean": float(np.mean(mag)),
                "std": float(np.std(mag)),
            }
        else:
            stats.update({
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            })

            if np.issubdtype(v.dtype, np.integer):
                info = np.iinfo(v.dtype)
                full_range = info.max - info.min
                used_range = stats["max"] - stats["min"]
                stats["int_range"] = {
                    "type_min": int(info.min),
                    "type_max": int(info.max),
                    "used_min": stats["min"],
                    "used_max": stats["max"],
                    "used_percent": float(100 * used_range / full_range)
                }

        self.data[name] = stats

    def export(self, path=None):
        if path:
            with open(path, "w") as f:
                json.dump(self.data, f, indent=2)
        return self.data
