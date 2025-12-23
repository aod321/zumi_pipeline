from motor_interface import MotorDriver, MotorState


class DMMotorDriver(MotorDriver):
    """
    Adapter over DM_CAN to satisfy MotorDriver interface.
    """

    def __init__(self, serial_port: str, slave_id: int, master_id: int, logger=None,
                 baud: int = 921600, timeout: float = 0.1, auto_set_zero: bool = True):
        self.logger = logger

        try:
            from DM_CAN import DM_Motor_Type, Motor, MotorControl, Control_Type
            import serial
        except ImportError as exc:
            raise ImportError(f"DM_CAN or serial not available: {exc}")

        self.serial_port = serial_port
        self.slave_id = slave_id
        self.master_id = master_id

        self.ser = serial.Serial(self.serial_port, baud, timeout=timeout)
        if not self.ser.is_open:
            self.ser.open()

        self.motor = Motor(DM_Motor_Type.DMH3510, self.slave_id, self.master_id)
        self.ctrl = MotorControl(self.ser)
        self.ctrl.addMotor(self.motor)

        if not self.ctrl.switchControlMode(self.motor, Control_Type.MIT):
            raise RuntimeError(
                f"Motor init failed: check connection and ID config "
                f"(SlaveID={hex(slave_id)}, MasterID={hex(master_id)}, Port={serial_port})"
            )

        self.enable()
        if auto_set_zero:
            self.set_zero()

    def enable(self):
        self.ctrl.enable(self.motor)

    def disable(self):
        try:
            self.ctrl.disable(self.motor)
        except Exception:
            pass

    def set_zero(self):
        self.ctrl.set_zero_position(self.motor)

    def command(self, torque: float, position: float = 0.0, velocity: float = 0.0, kp: float = 0.0, kd: float = 0.0):
        self.ctrl.controlMIT(self.motor, position, velocity, kp, kd, torque)

    def get_state(self) -> MotorState:
        return MotorState(
            position=float(self.motor.getPosition()),
            velocity=float(self.motor.getVelocity()),
            torque=float(self.motor.getTorque()),
        )

    def shutdown(self):
        self.disable()
        if getattr(self, "ser", None) and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
