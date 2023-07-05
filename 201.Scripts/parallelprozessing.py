
import multiprocessing
import pickle
import os
import sys
import time
from threading import Thread


class ProcRunningError(ChildProcessError):

    def __init__(self, msg):
        super(ProcRunningError, self).__init__(msg)


class ProcHandler(object):

    def __init__(self, proc, para, on_complete: None):
        self.proc = proc
        self.para = para
        self.is_running = False
        self.result = None
        self.t = None
        self.child_pid = None
        self.on_complete = on_complete

    def handler(self, pipe_r, pid, pipe_w):
        p_size = 16384
        self.child_pid = pid
        r = os.read(pipe_r, p_size)
        result = r
        while len(r) >= p_size:
            r = os.read(pipe_r, p_size)
            result += r
        self.result = pickle.loads(result)
        self.child_pid = None
        self.t = None
        self.is_running = False
        os.close(pipe_r)
        os.close(pipe_w)

    def run_on_complete(self):
        if self.on_complete is not None:
            self.on_complete()

    def start(self):
        if self.is_running:
            raise ProcRunningError("Process is already running!")
        self.is_running = True
        pipe_r, pipe_w = os.pipe()
        pid = os.fork()
        if pid < 1:  # if is child
            data = pickle.dumps(self.proc(*self.para))
            os.write(pipe_w, data)
            sys.exit(0)
        else:  # if is parend
            self.t = Thread(target=self.handler, args=(pipe_r, pid, pipe_w))
            self.t.start()

    def reset(self, para):
        if self.is_running:
            raise ProcRunningError("Can't reset running Process!")
        self.para = para
        self.result = None


def run_parallel(proc, max_subproc_count=multiprocessing.cpu_count()*2, on_block_complete=lambda n, c: print(n, "/", c)):
    out = [None] * len(proc)
    handler = []
    index = 0
    for i in range(min(max_subproc_count, len(proc))):
        func, para = proc[index]
        h = ProcHandler(func, para)
        handler.append((h, index))
        h.start()
        index += 1
    while index < len(proc):
        try:
            os.wait()  # wait for a proc to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(min(max_subproc_count, len(proc))):
            h, hi = handler[i]
            if not h.is_running:
                out[hi] = h.result
                h.run_on_complete()
                func, para = proc[index]
                h = ProcHandler(func, para, lambda : on_block_complete(index, len(proc)))
                handler[i] = (h, index)
                h.start()
                index += 1
                if index < len(proc):
                    break
    left = min(max_subproc_count, len(proc))
    blocked = []
    while left > 0:
        try:
            os.wait()  # wait for a proc to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(min(max_subproc_count, len(proc))):
            h, hi = handler[i]
            if (not h.is_running) and (not blocked.__contains__(hi)):
                out[hi] = h.result
                h.run_on_complete()
                left -= 1
                blocked.append(hi)
    return out


def run_mono(proc, parameters, max_subproc_count=multiprocessing.cpu_count()*2, on_block_complete=lambda c: c):
    out = [None] * len(parameters)
    handler = []
    index = 0
    for i in range(min(max_subproc_count, len(parameters))):
        para = parameters[index]
        h = ProcHandler(proc, para, lambda : on_block_complete(len(parameters)))
        handler.append((h, index))
        h.start()
        index += 1
    while index < len(parameters):
        try:
            os.wait()  # wait for a proc to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(min(max_subproc_count, len(parameters))):
            h, hi = handler[i]
            if not h.is_running:
                out[hi] = h.result
                try:
                    para = parameters[index]
                    h.reset(para)
                    h.run_on_complete()
                    handler[i] = (h, index)
                    h.start()
                    index += 1
                except IndexError:
                    break
    left = min(max_subproc_count, len(parameters))
    blocked = []
    while left > 0:
        try:
            os.wait()  # wait for a proc to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(min(max_subproc_count, len(parameters))):
            h, hi = handler[i]
            if (not h.is_running) and (not blocked.__contains__(hi)):
                out[hi] = h.result
                h.run_on_complete()
                left -= 1
                blocked.append(hi)
    return out
