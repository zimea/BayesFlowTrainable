import sys

class Logger(object):
    def __init__(self, logfile, out=sys.stdout, log_console=True):
        self.terminal = out
        self.logfile = logfile
        self.log = open(logfile, "w")
        self.log_console = log_console
   
    def write(self, message):
        if self.log_console:
            self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)  

    def flush(self):
        pass

    def pause(self):
        self.log.close()

    def resume(self):
        self.log = open(self.logfile, "a")
