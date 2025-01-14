import numpy as np

######################
# AUXILIARY FUNCTIONS:
######################


def gaussian_fn(maximum: float, sigma: float, multiple_of_sigma: int) -> np.ndarray:
    length = int(multiple_of_sigma * sigma)
    mu = int(np.floor(length / 2))
    t = np.linspace(0, length - 1, length)
    gaussian = maximum * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gaussian]


def gaussian_derivative_fn(
    gauss_A: float, drag: float, sigma: float, multiple_of_sigma: int
) -> np.ndarray:
    length = int(multiple_of_sigma * sigma)
    mu = int(np.floor(length / 2))
    t = np.linspace(0, length - 1, length)
    gaussian = gauss_A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    gaussian_derivative = drag * gaussian * (t - mu) / (sigma ** 2)
    return gaussian_derivative


def IQ_imbalance(gain: float, phase: float) -> list[float]:
    cos = np.cos(phase)
    sin = np.sin(phase)
    coeff = 1 / ((1 - gain ** 2) * (2 * cos ** 2 - 1))
    matrix = [(1 - gain) * cos, (1 + gain) * sin, (1 - gain) * sin, (1 + gain) * cos]
    correction_matrix = [float(coeff * x) for x in matrix]
    return correction_matrix


########################################################################################
##############################           ELEMENTS         ##############################
########################################################################################
# readout pulse
rr_time_of_flight = 1600  # in ns
rr_LO = int(9.453e9)
rr_IF = int(-49.4e6)

# qubit pulse
# g-e transition, dodn't delete below commented codes
qubit_LO = int(4.1235e9)
qubit_IF = int(-47.55e6)

# qubit_LO = 4.0e9
# qubit_IF = -47.55e6


rr_mixer_gain = 0.048832416534423814
rr_mixer_phase = -0.10307483673095706
rr_offset_I = 0.10078125
rr_offset_Q = -0.008203125

qubit_mixer_gain = 0.01411627754569054
qubit_mixer_phase = 0.07736417464911938
qubit_offset_I = -0.011938131041824819
qubit_offset_Q = -0.0015285410918295388


qubit_mixer_offsets = {
    "I": qubit_offset_I,
    "Q": qubit_offset_Q,
    "G": qubit_mixer_gain,
    "P": qubit_mixer_phase,
}

rr_mixer_offsets = {
    "I": rr_offset_I,
    "Q": rr_offset_Q,
    "G": rr_mixer_gain,
    "P": rr_mixer_phase,
}

########################################################################################
##############################            PULSES          ##############################
########################################################################################

################################### ARBITRARY PULSES ###################################
# readout pulse
readout_pulse_len = 4000
readout_pulse_amp = 0.2

# qubit pulse
# CW pulse
cw_pulse_len = 2000
cw_pulse_amp = 0.25
# saturation pulse
saturation_pulse_len = 4000
saturation_pulse_amp = 0.25

# (maximum, sigma, multiple_of_sigma)
gaussian_pulse_wf_I_samples = gaussian_fn(0.25, 60, 4)
gaussian_pulse_len = len(gaussian_pulse_wf_I_samples)

# (maximum, drag, sigma, multiple_of_sigma)
gaussian_derivative_wf_samples = gaussian_derivative_fn(0.25, 1, 60, 4)
gaussian_drag_pulse_len = len(gaussian_derivative_wf_samples)

################################### EXCLUSIVE PULSES ###################################
# qubit square pi and pi2 pulses
sq_pi_len = 512  # must be an integer multiple of 4 >= 16
sq_pi2_len = 256  # must be an integer multiple of 4 >= 16
sq_pi_amp = 0.25 * 1.445
sq_pi2_amp = 0.25 * 1.445
# Notice:
# Here the initial voltage has been set as the maximum 0.5, double of 0.25
# In the script, the square pi pulse relative amplitude ascale has to be set as 1

# qubit gaussian pi pulse
gauss_pi_amp = 0.25 * 1.8  # 1.846
gauss_pi_sigma = 125  # 120  160
gauss_pi_chop = 4
gauss_pi_samples = gaussian_fn(gauss_pi_amp, gauss_pi_sigma, gauss_pi_chop)
gauss_pi_len = len(gauss_pi_samples)

# DRAG correction
drag_coeff = 47
gauss_pi_drag_samples = gaussian_derivative_fn(
    gauss_pi_amp, drag_coeff, gauss_pi_sigma, gauss_pi_chop
)

# qubit gaussian pi2 pulse
gauss_pi2_amp = 0.25 * 0.9  # minus 0.02
gauss_pi2_sigma = 125
gauss_pi2_chop = 4
gauss_pi2_samples = gaussian_fn(gauss_pi2_amp, gauss_pi2_sigma, gauss_pi2_chop)
gauss_pi2_len = len(gauss_pi2_samples)

drag_coeff_pi2 = 47
gauss_pi2_drag_samples = gaussian_derivative_fn(
    gauss_pi2_amp, drag_coeff_pi2, gauss_pi2_sigma, gauss_pi2_chop
)

########################################################################################
################################           PORTS         ###############################
########################################################################################

qubit_ports = {"I": 4, "Q": 3}  # from OPX's point of view, these are analog outputs
rr_ports = {"I": 1, "Q": 2, "out": 1}  # "out" is analog input from the OPX's POV

########################################################################################
###############################           CONFIG         ###############################
########################################################################################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": rr_offset_I},  # RR I
                2: {"offset": rr_offset_Q},  # RR Q
                3: {"offset": qubit_offset_Q},  # qubit Q
                4: {"offset": qubit_offset_I},  # qubit I
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {},
            "analog_inputs": {
                rr_ports["out"]: {"offset": 0.0},
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", qubit_ports["I"]),
                "Q": ("con1", qubit_ports["Q"]),
                "lo_frequency": int(qubit_LO),
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": int(qubit_IF),
            "operations": {
                "CW": "CW_pulse",
                "pi": "pi_pulse",
                "pi_drag": "pi_drag_pulse",
                "pi2_drag": "pi2_drag_pulse",
                "pi2": "pi2_pulse",
                "sqpi": "sqpi_pulse",
                "sqpi2": "sqpi2_pulse",
                "gaussian": "gaussian_pulse",
                "drag": "gaussian_drag_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "rr": {
            "mixInputs": {
                "I": ("con1", rr_ports["I"]),
                "Q": ("con1", rr_ports["Q"]),
                "lo_frequency": int(rr_LO),
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": int(rr_IF),
            "operations": {
                "CW": "CW_pulse",
                "gaussian": "gaussian_pulse",
                "readout": "readout_pulse",
            },
            "outputs": {"out1": ("con1", rr_ports["out"])},
            "time_of_flight": rr_time_of_flight,
            "smearing": 12,
        },
    },
    "pulses": {
        "CW_pulse": {
            "operation": "control",
            "length": cw_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": gauss_pi_len,
            "waveforms": {"I": "gauss_pi_wf", "Q": "zero_wf"},
        },
        "pi2_pulse": {
            "operation": "control",
            "length": gauss_pi2_len,
            "waveforms": {"I": "gauss_pi2_wf", "Q": "zero_wf"},
        },
        "pi_drag_pulse": {
            "operation": "control",
            "length": gauss_pi_len,
            "waveforms": {"I": "gauss_pi_wf", "Q": "gauss_pi_drag_wf"},
        },
        "pi2_drag_pulse": {
            "operation": "control",
            "length": gauss_pi2_len,
            "waveforms": {"I": "gauss_pi2_wf", "Q": "gauss_pi2_drag_wf"},
        },
        "sqpi_pulse": {
            "operation": "control",
            "length": sq_pi_len,
            "waveforms": {"I": "sq_pi_pi_wf", "Q": "zero_wf"},
        },
        "sqpi2_pulse": {
            "operation": "control",
            "length": sq_pi2_len,
            "waveforms": {"I": "sq_pi_pi2_wf", "Q": "zero_wf"},
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_pulse_len,
            "waveforms": {"I": "saturation_wf", "Q": "zero_wf"},
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "gaussian_drag_pulse": {
            "operation": "control",
            "length": gaussian_drag_pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "gauss_derivative_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
                "optW1": "optW1",
                "optW2": "optW2",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "saturation_wf": {
            "type": "constant",
            "sample": saturation_pulse_amp,
        },
        "const_wf": {
            "type": "constant",
            "sample": cw_pulse_amp,
        },
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "gauss_wf": {
            "type": "arbitrary",
            "samples": gaussian_pulse_wf_I_samples,
        },
        "gauss_derivative_wf": {
            "type": "arbitrary",
            "samples": gaussian_derivative_wf_samples,
        },
        "readout_wf": {
            "type": "constant",
            "sample": readout_pulse_amp,
        },
        "gauss_pi_wf": {
            "type": "arbitrary",
            "samples": gauss_pi_samples,
        },
        "gauss_pi_drag_wf": {
            "type": "arbitrary",
            "samples": gauss_pi_drag_samples,
        },
        "gauss_pi2_drag_wf": {
            "type": "arbitrary",
            "samples": gauss_pi2_drag_samples,
        },
        "gauss_pi2_wf": {
            "type": "arbitrary",
            "samples": gauss_pi2_samples,
        },
        "sq_pi_pi_wf": {
            "type": "constant",
            "sample": sq_pi_amp,
        },
        "sq_pi_pi2_wf": {
            "type": "constant",
            "sample": sq_pi2_amp,
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "integW1": {
            "cosine": [1.0] * int(readout_pulse_len / 4),
            "sine": [0.0] * int(readout_pulse_len / 4),
        },
        "integW2": {
            "cosine": [0.0] * int(readout_pulse_len / 4),
            "sine": [1.0] * int(readout_pulse_len / 4),
        },
        "optW1": {
            "cosine": [1.0] * int(readout_pulse_len / 4),
            "sine": [0.0] * int(readout_pulse_len / 4),
        },
        "optW2": {
            "cosine": [0.0] * int(readout_pulse_len / 4),
            "sine": [1.0] * int(readout_pulse_len / 4),
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": int(qubit_IF),
                "lo_frequency": int(qubit_LO),
                "correction": IQ_imbalance(
                    qubit_mixer_offsets["G"], qubit_mixer_offsets["P"]
                ),
            },
        ],
        "mixer_rr": [
            {
                "intermediate_frequency": int(rr_IF),
                "lo_frequency": int(rr_LO),
                "correction": IQ_imbalance(
                    rr_mixer_offsets["G"], rr_mixer_offsets["P"]
                ),
            }
        ],
    },
}
