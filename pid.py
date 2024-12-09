import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-1, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

    def update(self, measured_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0.0:
            dt = 1e-6  

        error = self.setpoint - measured_value
        proportional = self.Kp * error

        self._integral += error * dt
        integral = self.Ki * self._integral

        derivative = self.Kd * (error - self._prev_error) / dt

        output = proportional + integral + derivative

        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        self._prev_error = error
        self._last_time = current_time

        return output

    def reset(self):
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

