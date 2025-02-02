"""
A python class describing a power rabi measurement using the QM 
system. 
This class serves as a QUA script generator with user-defined parameters. It 
also defines how the information is retrieved from result handles.
"""
# --------------------------------- Imports ------------------------------------
from qcrew.experiments.coax_test.imports import *
from qcrew.experiments.coax_test.measurements.v5.BaseExperiment_newlibs import *

stage_module_path = resolve_name(".stage", "qcrew.experiments.coax_test.imports")
if stage_module_path not in sys.modules:
    import qcrew.experiments.coax_test.imports.stage as stg
else:
    reload(stg)
# ---------------------------------- Class -------------------------------------


class PowerRabi(Experiment):
    def __init__(self, qubit_op, readout_op, fit_fn = None, **other_params):

        # Get attributes
        self.qubit_op = qubit_op
        self.readout_op = readout_op
        self.fit_fn = fit_fn

        # Passes remaining parameters to parent
        super().__init__(**other_params)

    def QUA_play_pulse_sequence(self):
        """
        Defines pulse sequence to be played inside the experiment loop
        """
        """
        self.qubit.play(self.qubit_op, self.x)
        align(self.qubit.name, self.rr.name)
        self.rr.measure(self.readout_op)  # This should account for intW
        wait(int(self.wait_time // 4), self.qubit.name)
        self.QUA_save_results_to_stream()
        """
        play(self.qubit_op * amp(self.x), "qubit")
        align("qubit", "rr")
        measure(
            self.readout_op * amp(0.2),
            "rr",
            None,
            demod.full("integW1", self.I),
            demod.full("integW2", self.Q),
        )
        wait(int(self.wait_time // 4), "qubit")
        save(self.x, self.x_stream)
        save(self.I, self.I_stream)
        save(self.Q, self.Q_stream)


# -------------------------------- Execution -----------------------------------


if __name__ == "__main__":

    #############        SETTING EXPERIMENT      ################

    qubit = stg.qubit  # TODO I'm assuming smth like this to get mode objects
    rr = stg.rr

    exp_params = {
        "reps": 1000,  # number of sweep repetitions
        "wait_time": 32000,  # delay between reps in ns, an integer multiple of 4 >= 16
        "x_sweep": (-1.9, 1.9, 0.1),  # x sweep is set by start, stop, and step
        "is_x_sweep_arbitrary": False  # Tells whether x sweep has arbitrary values 
        "qubit_op": "pi",  # Operations to be used in the exp.
        "readout_op": "readout",
        "fit_fn": "sine",  # name eof the fit function
    }

    SAMPLE_NAME = "coax_a"
    EXP_NAME = "power_rabi"
    PROJECT_FOLDER_NAME = "coax_test"
    DATAPATH = Path.cwd() / "data"

    experiment = PowerRabi(exp_params)
    power_rabi = experiment.QUA_sequence()

    ###################        RUN MEASUREMENT        ############################

    job = stg.qm.execute(power_rabi)

    ####################        INVOKE HELPERS        ###########################
    # fetch helper and plot hepler
    fetcher = Fetcher(handle=job.result_handles, num_results=experiment.reps)
    plotter = Plotter(title=EXP_NAME, xlabel="Amplitude scale factor")
    stats = (None, None, None)  # to hold running stats (stderr, mean, variance * (n-1))

    # initialise database under dedicated folder
    db = initialise_database(
        exp_name=EXP_NAME,
        sample_name=SAMPLE_NAME,
        project_name=PROJECT_FOLDER_NAME,
        path=DATAPATH,
        timesubdir=False,
        timefilename=True,
    )

    ##################        LIVE POST-PROCESSING LOOP        ####################

    with DataSaver(db) as datasaver:

        ###############        SAVE MEASUREMENT RUN METADATA       ###################

        datasaver.add_metadata(exp_params)

        while fetcher.is_fetching:
            ###############            FETCH PARTIAL RESULTS         ###############

            (num_so_far, update_results) = fetcher.fetch()
            if (
                not update_results
            ):  # empty dict return means no new results are available
                continue
            ################            LIVE SAVE RESULTS         ##################
            datasaver.update_multiple_results(
                update_results, save=["I", "Q"], group="data"
            )

            ########            CALCULATE RUNNING MEAN STANDARD ERROR         #########

            ys_raw = np.sqrt(update_results["Y_SQ_RAW"])
            ys_raw_avg = np.sqrt(update_results["Y_SQ_RAW_AVG"])
            stats = get_std_err(ys_raw, ys_raw_avg, num_so_far, *stats)

            #############            LIVE PLOT AVAILABLE RESULTS         ###########

            ys = np.sqrt(update_results["Y_AVG"])  # latest batch of average signal
            xs = update_results["X"]
            plotter.live_plot(
                xs, ys, num_so_far, fit_fn=experiment.fit_fn, err=stats[0]
            )
            time.sleep(1)  # prevent over-fetching, over-saving, ulta-fast live plotting

        ##################         SAVE REMAINING DATA         #####################

        # to save final average and sweep variables, we extract them from "update_results"
        final_save_dict = {"Y_AVG": ys, "X": xs}
        datasaver.add_multiple_results(
            final_save_dict, save=["Y_AVG", "X"], group="data"
        )

    ##########################          fin           #############################

    print(job.execution_report())
