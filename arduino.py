import serial

class BoardController:
    def __init__(self, port = '/dev/ttyAcM0', baud_rate = 9600):
        self.ser = serial.Serial(port, baud_rate)
        self.ser.reset_input_buffer()

    def short_sleeve(self):
        self.ser.write('1'.encode())

    def long_sleeve(self):
        self.ser.write('2'.encode())

    def trousers(self):
        self.ser.write('3'.encode())

    def maintenance(self, panel, action):
        # panel: north, south, east, west, extraSouth
        # action: open, close
        self.ser.write(f'{action}{panel}'.encode())

