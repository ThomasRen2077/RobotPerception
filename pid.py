import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-1, 1)):
        """
        Initializes the PID controller.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired value to achieve.
            output_limits (tuple): Minimum and maximum limits for the output.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        # Internal variables
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

    def update(self, measured_value):
        """
        Computes the PID controller output.

        Args:
            measured_value (float): Current value being measured.

        Returns:
            float: Controller output (e.g., input voltage).
        """
        # Compute time delta
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0.0:
            dt = 1e-6  # Avoid division by zero

        # Compute error
        error = self.setpoint - measured_value

        # Proportional term
        proportional = self.Kp * error

        # Integral term
        self._integral += error * dt
        integral = self.Ki * self._integral

        # Derivative term
        derivative = self.Kd * (error - self._prev_error) / dt

        # Compute output
        output = proportional + integral + derivative

        # Apply output limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))

        # Scale output by the time step (dt)
        scaled_output = output * dt

        # Update stored variables
        self._prev_error = error
        self._last_time = current_time

        return scaled_output

    def reset(self):
        """
        Resets the PID controller's internal state.
        """
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

