from qcrew.experiments.coax_test.imports import *  # Opens QM
from abc import abstractmethod
import numpy as np


class Experiment:
    """
    Abstract class for experiments using QUA sequences.
    """

    def __init__(
        self,
        reps,
        wait_time,
        x_sweep,
        is_x_explicit,
        y_sweep=None,
        is_y_explicit=None,
        extra_QUA_var=None,
    ):

        # Experiment loop variables
        self.reps = reps
        self.wait_time = wait_time

        # Sweep configurations
        self.sweep_config = {"n": (0, self.reps, 1), "x": x_sweep, "y": y_sweep}
        self.is_sweep_explicit = {"n": False, "x": is_x_explicit, "y": is_y_explicit}

        # QUA variable definitions {name:type}
        self.QUA_var_list = {
            "n": int,
            "x": None,  # defined in _check_sweeps
            "y": None,
            "I": fixed,
            "Q": fixed,
        }

        # List of variables to be sent to streams. More can be added in _check_sweeps
        self.QUA_stream_list = ['I', 'Q']

        # Set attributes for QUA variables (specified in QUA_variable_declaration)
        for var_name in self.QUA_var_list.keys():
            setattr(self, var_name, None)

        # Result tags for stream processing experiments
        self.X_tag = "X"
        self.Y_tag = "Y"
        self.I_tag = "I"
        self.Q_tag = "Q"
        self.Z_SQ_RAW_tag = "Z_SQ_RAW"
        self.Z_SQ_RAW_AVG_tag = "Z_SQ_RAW_AVG"
        self.Z_AVG_tag = "Z_AVG"

    def _check_sweeps(self):
        """
        Check if each x and y sweep contains numeric elements and has the
        information needed by qua.for_ or qua.for_all_ loops
        """

        # Check if the values of sweep are numeric. Includes numpy floats.
        is_sweep_numeric = all(isinstance(n, (int, float)) for n in new_sweep)
        if not is_sweep_numeric:
            print("error: all sweep values should be numeric")
            raise SystemExit("Unable to create Experiment")

        # Assigns new_sweep to x or y dimension. Doesn't expect more than that
        if not hasattr(self, "_sweeps"):
            sweep_dim = "x"
            self._sweeps = dict()
        else:
            sweep_dim = "y"

        # Check if sweep is linear or arbitrary
        if isinstance(new_sweep, tuple) and len(new_sweep) == 3:
            # new_sweep is linear
            is_arbitrary = False
        elif isinstance(new_sweep, (list, np.ndarray)):
            # new_sweep is arbitrary
            is_arbitrary = True
        else:
            # new_sweep is not well defined
            print("error: sweep %s is not well defined" % sweep_dim)
            raise SystemExit("Unable to create Experiment")

        new_sweep_dict = {"arbitrary": is_arbitrary, "vals": new_sweep}
        self._sweeps[sweep_dim] = new_sweep_dict

        return

    def QUA_sweep(self, sweep_dim, QUA_function):

        # Get sweep variable
        sweep_var = getattr(self, sweep_dim)

        if sweep_var == None:
            # If sweep is not configured, simply play function 
            QUA_function()

        else:
            # Check the type of the loop
            if self.is_sweep_explicit[sweep_dim]:
                # Wrap function in qua.for_ loop
                start, stop, step = self.sweep_config[sweep_dim]
                with for_(sweep_var, start, sweep_var < stop + step/2, sweep_var + step):
                    QUA_function()
            else:
                # Get array of values to sweep over
                loop_array = self.sweep_config[sweep_dim]
                # Wrap function in qua.for_all_ loop
                with for_all_(sweep_var, loop_array):
                    QUA_function()

    @abstractmethod
    def QUA_play_pulse_sequence(self):
        """
        Macro that defines the QUA pulse sequence inside the experiment loop. It is
        specified by the experiment (spectroscopy, power rabi, etc.) in the child class.
        """
        pass

    def QUA_sequence(self):
        """
        Method that returns the QUA sequence to be executed in the quantum machine.
        """

        # Check if the sweep configurations are sane
        self._check_sweeps()

        with program() as qua_sequence:

            # Initial variable and stream declarations
            self.QUA_variable_declaration()
            self.QUA_stream_declaration()

            # Experiment loop
            self.QUA_sweep(
                "n", self.QUA_sweep(
                    "x", self.QUA_sweep(
                        "y", self.QUA_play_pulse_sequence()
                    )
                )
            )

            # Define stream processing
            self.QUA_stream_processing()

        return qua_sequence

    def QUA_variable_declaration(self):
        """
        Macro that calls QUA variable declaration statements. The variables are
        specified in QUA_var_list.
        """
        for key, val in self.QUA_var_list:
            if val:
                setattr(self, key, declare(val))

    def QUA_stream_declaration(self):
        """
        Macro that calls QUA stream declaration statements. The variables are
        specified in QUA_stream_list.
        """
        for key, val in self.QUA_stream_list:
            if val:
                setattr(self, key, declare_stream())

    def QUA_save_results_to_stream(self):
        """
        Macro that calls QUA save statements.
        """
        save(self.x, self.x_stream)
        save(self.I, self.I_stream)
        save(self.Q, self.Q_stream)

    def QUA_stream_processing(self):
        """
        Macro that calls QUA save statements. QUA variables x, I, Q and respective
        streams are defined in method QUA_variable_declaration
        """
        with stream_processing():
            I_raw = self.I_stream.buffer(self.x_sweep_len)
            Q_raw = self.Q_stream.buffer(self.x_sweep_len)  # to reshape result streams
            I_avg = I_raw.average()
            Q_avg = Q_raw.average()  # to get running averages

            I_raw.save_all(self.I_tag)
            Q_raw.save_all(self.Q_tag)  # to save all raw I and Q data

            # we need these two streams to calculate std err in a single pass
            (I_raw * I_raw + Q_raw * Q_raw).save_all(self.Z_SQ_RAW_tag)
            (I_raw * I_raw + Q_raw * Q_raw).average().save_all(self.Z_SQ_RAW_AVG_tag)

            # to live plot latest average
            (I_avg * I_avg + Q_avg * Q_avg).save(self.Z_AVG_tag)
            self.x_stream.buffer(self.x_sweep_len).save(self.X_tag)  # sweep variable
