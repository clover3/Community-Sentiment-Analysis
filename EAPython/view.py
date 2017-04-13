import numpy as np

root_dir = "debugs"


class Logger:
    def __init__(self):
        self.unit_dict = dict()
        np.set_printoptions(precision=5, suppress=True)

    def print(self, name, obj):
        if name not in self.unit_dict:
            self.unit_dict[name] = LoggingUnit(name)

        self.unit_dict[name].print(obj)


class LoggingUnit:
    def __init__(self, name):
        self.f = open(root_dir + "\\" + name, "w+")
        self.id = 0
        self.f.write("File Beggining --------------------\n")

    def print(self, obj):
        self.log(obj.__str__())

    def log(self, msg):
        self.f.write("------------Log {}----------------\n".format(self.id))
        self.f.write(msg + "\n----------------------------------\n")
        self.id += 1




