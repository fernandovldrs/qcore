# import all objects defined in the __init__.py file in the 'imports' folder
from qcrew.experiments.coax_test.imports import *
reload(cfg), reload(stg)  # reloads modules before executing the code below

# NOTE: make changes to lo, if, tof, mixer offsets in 'configuration.py'
# NOTE: make changes to constant pulse amp and pulse duration in the qua script below

# import Signal Hound Spectrum Analyzer
from qcrew.codebase.instruments import Sa124

MEAS_NAME = "spectrum_analysis"  # used for naming the saved data file

########################################################################################
########################           MEASUREMENT SEQUENCE         ########################
########################################################################################

# initialize the SA
sa = Sa124(name="sa", serial_number=19184645)

# define quantum element whose signal will be analyzed. The quantum element is required
# to have a CW operation defined.
q_elem = stg.rr
q_elem_name = q_elem._name

# define center frequency
center = q_elem.parameters["lo_freq"]

# define frequency span in hertz
span = abs(3 * q_elem.parameters["int_freq"])

# define reference power in dBm
ref_power = 0

# QUA script
with program() as cw:
    with infinite_loop_():
        play("CW", q_elem_name)

########################################################################################
############################           GET RESULTS         #############################
########################################################################################
job = stg.qm.execute(cw)
freqs, amps = sa.sweep(center=center, span=span, ref_power=ref_power)
job.halt()
sa.disconnect()
plt.plot(freqs, amps)

########################################################################################
############################           SAVE RESULTS         ############################
########################################################################################

metadata = f"{q_elem_name = }, {center = }, {span = }, {ref_power = }"
filename = f"{datetime.now().strftime('%H-%M-%S')}_{MEAS_NAME}"
datapath = DATA_FOLDER_PATH / (filename + ".csv")
imgpath = DATA_FOLDER_PATH / (filename + ".png")

with datapath.open("w") as f:
    f.write(metadata)
    np.savetxt(datapath, [freqs, amps], delimiter=",")
plt.savefig(imgpath)

########################################################################################
########################################################################################
########################################################################################
plt.show()  # this blocks execution, and is hence run at the end of the script
