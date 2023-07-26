import multiprocessing
import pickle
import os
import sys
import time
from threading import Thread


class ProcRunningError(ChildProcessError):
    """
    Just a custom Exeption, witch should not be needed.

    Triggers when a already running Process is called to start.
    """

    def __init__(self, msg):
        super(ProcRunningError, self).__init__(msg)


class ProcHandler(object):
    """
    This Class Handeles the Data and communication for
    a single Subprocess.
    """

    def __init__(self, func, para, on_complete: None):
        self.function = func  # The Function that the Subprocess will run
        self.parameter = para  # The Parameters to run the Function with
        self.is_running = False
        self.result = None  # Buffer for the Result returned by the Subprocess
        self.thread = None  # Tread used to wait for the result
        self.child_pid = None  # PID of the Subprocess
        self.on_complete = on_complete  # Optional Function to call when the Subprocess is Done

    def handler(self, pipe_r, pid, pipe_w):
        """
        This Method handeles the communication with the Subprocess.
        aka it waits till the Subprocess writes its Result to the Pipe.

        :param pipe_r:  The Read-End of the Pipe used to send the Result
        :param pid:     The PID of the Subprocess
        :param pipe_w:  The Write-End of the Pipe (needed to delete the Pipe after)
        """
        p_size = 16384  #~16kb: byte size to read fom the Pipe
        self.child_pid = pid  # Set PID of Subprocess

        # Start Reading the Result (will also wait for it)
        r = os.read(pipe_r, p_size)
        result = r
        while len(r) >= p_size:
            r = os.read(pipe_r, p_size)
            result += r
        self.result = pickle.loads(result)  # decode the bytestring result into usable Data.

        # reset Subprocess Info
        self.child_pid = None
        self.thread = None
        self.is_running = False

        # remove unused Pipe
        os.close(pipe_r)
        os.close(pipe_w)

    def run_on_complete(self):
        if self.on_complete is not None:
            self.on_complete()

    def start(self):
        """
        This Function will start a Subprocess
        """
        if self.is_running:  # can't start what is already running
            raise ProcRunningError("Process is already running!")
        self.is_running = True

        pipe_r, pipe_w = os.pipe()  # open a Pipe for communication
        pid = os.fork()  # fork the Process to make Subprocess with identical Memory (there is no shared Memory)

        if pid < 1:  # if is Subprocess
            # run the Function with the given Parameters and return result via Pipe
            data = pickle.dumps(self.function(*self.parameter))
            os.write(pipe_w, data)
            sys.exit(0) # Then exit, witch will signal the Main-Process
        else:  # if is Main-Process
            # start Tread to wait for the result
            self.thread = Thread(target=self.handler, args=(pipe_r, pid, pipe_w))
            self.thread.start()

    def reset(self, para):
        """
        This Method will reset The Handler and prepare it for the next run.

        :param para: new Parameters for the next Run
        """
        if self.is_running:
            raise ProcRunningError("Can't reset running Process!")
        self.parameter = para
        self.result = None


def run_parallel(proc, parameters, max_subproc_count=multiprocessing.cpu_count() * 2, on_block_complete=lambda c: c):
    """
    :param proc: Function to run multiple Instances in Parallel
    :param parameters: List of Parameter Tuples for the Function
    :param max_subproc_count: Max count of Subprocesses running at the same time. def=Core_count*2
    :param on_block_complete: Optinal Function to run when a Subprocess is done. (usefull for progressbars)
    :return: List containing all results off the Subprocceses in same order to the Parameters-List
    """
    results = [None] * len(parameters)  # List of Results that will be returned later, initalized as list of None-Values
    handler_list = []  # List of Subprocess handlers (class ProcHandler)
    index = 0  # index of Parameter-Tuple to run next
    for i in range(min(max_subproc_count, len(parameters))):  # initalize handlers with max_count or Subprocesscount if less.
        para = parameters[index]
        handler = ProcHandler(proc, para, lambda: on_block_complete(len(parameters)))
        handler_list.append((handler, index))
        handler.start()
        index += 1
    while index < len(parameters):  # as long as there are Parameter-Tuples left that wait for a Subprocess too finish and restart it with new Parameters
        try:
            os.wait()  # wait for a Subprocess to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(len(handler_list)):  # do throw all handler and if a Subprocess is Done restart with new Parameter
            handler, result_index = handler_list[i]
            if not handler.is_running:  # Subprocess is Done
                results[result_index] = handler.result  # enter Result into list
                try:  # restart Subprocess with new Paramete
                    para = parameters[index]
                    handler.reset(para)
                    handler.run_on_complete()
                    handler_list[i] = (handler, index)
                    handler.start()
                    index += 1
                except IndexError:  # no Parameter-Tuples left to run
                    break
    left_handler_count = len(handler_list)  # count of handler that are still running
    done_handlers = []  # list of handler that are know to be already done.
    while left_handler_count > 0:  # wait as long as there is a Subprocess is still running
        try:
            os.wait()  # wait for a Subprocess to be done
        except ChildProcessError:
            pass
        time.sleep(1)
        for i in range(len(handler_list)):  # check all handlers if there done
            handler, result_index = handler_list[i]
            if (not handler.is_running) and (not done_handlers.__contains__(result_index)):  # if a new Subprocess is done count that down
                results[result_index] = handler.result  # enter Result of that Subprocess int the list.
                handler.run_on_complete()
                left_handler_count -= 1
                done_handlers.append(result_index)
    # All the Parameter-Tuples have run throw, and all Results are now in the out list.
    return results
