import time

class MeasureTime():
    def __init__(self) -> None:
        self.count = 0
        self.total_time = 0
        self.start_time = 0
        self.end_time = 0
        self.running = False
        self.children = {}

        self.start()

    def add_child(self, name):
        if name in self.children.keys():
            if self.children[name].running:
                raise Exception(f"Profiler {name} is already running!")
            else:
                self.children[name].start()
                return
            
        for p in self.children.values():
            if p.running:
                p.add_child(name)
                return
            
        self.children[name] = MeasureTime()

    def start(self):
        self.start_time = time.time()
        self.running = True

    def stop(self):
        self.end_time = time.time()
        self.total_time += self.end_time - self.start_time
        self.count += 1
        self.running = False

    def get_average(self):
        return self.total_time / self.count
    
    def get_report(self):
        return {
            "head": (self.get_average(), self.total_time),
            "children": {name: p.get_report() for name, p in self.children.items()}
        }
    
    def stop_children(self, name):
        if not self.running:
            return False

        if name in self.children.keys():
            if self.children[name].running:
                self.children[name].stop()
                return True
            else:
                raise Exception(f"Profiler {name} is not running!")
        else:
            for key in self.children.keys():
                if self.children[key].stop_children(name):
                    return True
        return False

PROFILES = {}
LOG_FILE = "profiler.log"

def reset_profiles():
    global PROFILES
    PROFILES = {}

def start(name):
    if name in PROFILES.keys():
        if PROFILES[name].running:
            raise Exception(f"Profiler {name} is already running!")
        else:
            PROFILES[name].start()
            return 

    for p in PROFILES.values():
        if p.running:
            p.add_child(name)
            return 
        
    PROFILES[name] = MeasureTime()

def stop(name):
    if name in PROFILES.keys():
        PROFILES[name].stop()
        return 
    else:
        for key in PROFILES.keys():
            if PROFILES[key].stop_children(name):
                return
    raise Exception(f"Profiler {name} is not running!")

def get_perc_of_total(name):
    total = sum([profile.total_time for profile in PROFILES.values()])
    return PROFILES[name].total_time / total * 100

def recursive_report(profile, name, level, file):
    avg, total = profile["head"]
    print(f"{'|   '*level}{name.ljust(32)}    |> {avg:.3f}s", file=file)
    for child_name, child_profile in profile["children"].items():
        recursive_report(child_profile, child_name, level+1, file)

def make_report():
    with open(LOG_FILE, "w") as f:
        print("========================PROFILER REPORT========================", file=f)


        for name, profile in PROFILES.items():

            p_report = profile.get_report()
            avg, total = p_report["head"]
            percentage = f"{get_perc_of_total(name):.3f}".zfill(5)
            print(f"{name.ljust(32)}    {avg:.3f}s", file=f)

            for child_name, child_profile in p_report["children"].items():
                recursive_report(child_profile, child_name, 1, f)


        print("===============================================================", file=f)

class Profile:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        start(self.name)

    def __exit__(self, type, value, traceback):
        stop(self.name)