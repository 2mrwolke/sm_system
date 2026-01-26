import numpy as np
import scipy


class LTI:
    """LTI represents a class for linear time-invariant systems."""

    def __init__(self):  # -> None
        # Defining the types of filters and their corresponding methods
        self.system = dict()
        self.system.update(
            {
                "Custom": self.custom,
                "Low-pass 1st": self.lp_1,
                "Low-pass 2nd": self.lp_2,
                "High-pass 1st": self.hp_1,
                "High-pass 2nd": self.hp_2,
                "Band-pass 2nd": self.bp_2,
                "Notch 2nd": self.notch_2,
                "All-pass 1st": self.ap_1,
                "All-pass 2nd": self.ap_2,
                "-- NONE --": self.none,
            }
        )
        pass

    def _get_times(self, input, sr):
        """Generate a series of time values for input signal with sample rate sr."""
        times = np.arange(0, input.shape[-1]) / sr
        return times

    @staticmethod
    def to_callable(tf, sr):
        """Return a callable function which performs convolution on filter(tf) with input."""

        def convolve(inputs):
            times = np.arange(0, inputs.shape[-1]) / sr
            t, result, _ = scipy.signal.lsim(
                tf, U=inputs, T=times, interp=True
            )
            return result

        return convolve

    @staticmethod
    def to_scipy_tf(func):
        """A function decorator to transform the given system into a scipy TransferFunction."""

        def wrapper(*args, **kwargs):
            system = func(*args, **kwargs)
            tf = scipy.signal.TransferFunction(system[0], system[1])
            if "is_callable" in kwargs and kwargs.get("is_callable") is True:
                sr = kwargs["sr"]
                fun = LTI.to_callable(tf, sr)
                return fun
            else:
                if "verbose" in kwargs and kwargs["verbose"] is True:
                    print(
                        "LTI-instance returned: scipy.signal.TransferFunctionContinuous"
                    )
                return tf

        return wrapper

    @to_scipy_tf
    def custom(self, **kwargs):
        zeros = kwargs["zeros"]
        poles = kwargs["poles"]
        gain = kwargs["gain"]
        system = scipy.signal.ZerosPolesGain(zeros, poles, gain)
        tf = system.to_tf()
        system = [tf.num, tf.den]
        return system

    @to_scipy_tf
    def lp_1(self, **kwargs):
        wc = kwargs["wc"]
        system = [[1], [1 / wc, 1]]
        return system

    @to_scipy_tf
    def lp_2(self, **kwargs):
        wc = kwargs["wc"]
        q = kwargs["q"]
        system = [[wc * wc], [1, wc / q, wc * wc]]
        return system

    @to_scipy_tf
    def hp_1(self, **kwargs):
        wc = kwargs["wc"]
        system = [[1, 0], [1, wc]]
        return system

    @to_scipy_tf
    def hp_2(self, **kwargs):
        wc = kwargs["wc"]
        q = kwargs["q"]
        system = [[1, 0, 0], [1, wc / q, wc * wc]]
        return system

    @to_scipy_tf
    def bp_2(self, **kwargs):
        wc = kwargs["wc"]
        q = kwargs["q"]
        system = [[wc / q, 0], [1, wc / q, wc * wc]]
        return system

    @to_scipy_tf
    def notch_2(self, **kwargs):
        wc = kwargs["wc"]
        q = kwargs["q"]
        system = [[1, 0, wc * wc], [1, wc / q, wc * wc]]
        return system

    @to_scipy_tf
    def ap_1(self, **kwargs):
        wc = kwargs["wc"]
        system = [[1, -wc], [1, wc]]
        return system

    @to_scipy_tf
    def ap_2(self, **kwargs):
        wc = kwargs["wc"]
        q = kwargs["q"]
        system = [[1, -wc / q, wc * wc], [1, wc / q, wc * wc]]
        return system

    @to_scipy_tf
    def none(self, **kwargs):
        system = [[1], [1]]
        return system

    def to_FIR(self, filter, tabs=256, sr=48_000):
        filter = filter.to_discrete(dt=1 / sr)
        t, fir = scipy.signal.dimpulse(filter, n=tabs)
        return fir[0][:, 0]

    def convolve(self, signal, filter="Low-pass 1st", **kwargs):
        filter = self.system[filter](**kwargs)
        t, yout = scipy.signal.impulse(filter)
        h = yout[0]
        result = np.convolve(h, signal, mode="full")
        return result

    def __str__(self):
        return str(self.system.keys())
