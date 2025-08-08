import os

from utils import load_json, write_json


class LossManager:
    def __init__(self):
        self.values = {}

    def get(self, key, default=0):
        values = self.values[key].get("val", [])
        return values[-1] if values else default

    def update(self, key, value=None, new_entry=False):
        assert (isinstance(key, dict) and value is None) or (key and value is not None)
        update_dict = key if value is None else {key: value}
        for key, val in update_dict.items():
            self.values.setdefault(key, {})
            self.values[key].setdefault("val", [])
            self.values[key].setdefault("idx", [])

            if new_entry:
                self.values[key]["val"].append(val)
                self.values[key]["idx"].append(0)
                continue
            
            for state_key in ["val", "idx"]:
                if self.values[key][state_key]:
                    continue
                self.values[key][state_key].append(0)
                
            value_history = self.values[key]["val"]
            count_history = self.values[key]["idx"]
            last_val = value_history[-1] if value_history else 0
            idx = count_history[-1] if count_history else 0

            new_avg = last_val + (val - last_val) / (idx + 1)
            self.values[key]["val"][-1] = new_avg
            self.values[key]["idx"][-1] += 1

    def save_dict(self, epoch=0):
        values = {key: state["val"] for key, state in self.values.items()}
        num_samples = {key: state["idx"] for key, state in self.values.items()}
        return {"epoch": epoch, "values": values, "num_samples": num_samples}

    def save(self, path, epoch=0):
        write_json(path, self.save_dict(epoch))

    def update_path(self, path, epoch=0, num_values=1):
        if not os.path.exists(path):
            self.save(path, epoch)
            return
        load_dict = load_json(path)
        load_dict["epoch"] = epoch
        for key, state in self.values.items():
            if key not in load_dict["values"]:
                load_dict["values"][key] = []
                load_dict["num_samples"][key] = []
            load_dict["values"][key] = load_dict["values"][key][:epoch]
            load_dict["num_samples"][key] = load_dict["num_samples"][key][:epoch]
            load_dict["values"][key] += state["val"][-num_values:]
            load_dict["num_samples"][key] += state["idx"][-num_values:]
        write_json(path, load_dict)
