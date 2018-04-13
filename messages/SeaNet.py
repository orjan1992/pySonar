class SeanetDecode:

    data_buffer = bytearray(0)

    def add(self, bytes_to_add):
        self.data_buffer = buffer_add(self.data_buffer, bytes_to_add)

        status = 2

        while status == 2:
            self.data_buffer, status, data_packet = buffer_check(self.data_buffer)

        if status == 1:
            return data_packet
        else:
            return None


def valid_ascii_num(b):
    if 0x30 <= b <= 0x39:
        return True
    else:
        return False


def valid_ascii_hex_num(b):
    if valid_ascii_num(b) or 0x41 <= b <= 0x46:
        return True
    else:
        return False


def ascii_hex_to_bin(b):
    if valid_ascii_hex_num(b):

        if valid_ascii_num(b):
            return b - 0x30
        else:
            return b - 0x37
    return 0


def buffer_check(buffer_array):

    if len(buffer_array) > 0:
        ret = buffer_array

        buffer_remaining = len(buffer_array)
        buffer_cnt = 0
        status = -1

        for b in buffer_array:

            buffer_remaining -= 1

            if b == 0x40:
                if buffer_remaining >= 6:
                    if valid_ascii_hex_num(buffer_array[buffer_cnt + 1]):
                        if valid_ascii_hex_num(buffer_array[buffer_cnt + 2]):
                            if valid_ascii_hex_num(buffer_array[buffer_cnt + 3]):
                                if valid_ascii_hex_num(buffer_array[buffer_cnt + 4]):

                                    AsciiLen = ascii_hex_to_bin(buffer_array[buffer_cnt + 1]) * 0x1000
                                    AsciiLen = AsciiLen + ascii_hex_to_bin(
                                        buffer_array[buffer_cnt + 2]) * 0x100
                                    AsciiLen = AsciiLen + ascii_hex_to_bin(
                                        buffer_array[buffer_cnt + 3]) * 0x10
                                    AsciiLen = AsciiLen + ascii_hex_to_bin(buffer_array[buffer_cnt + 4])

                                    BinLen = buffer_array[buffer_cnt + 6] * 0x100
                                    BinLen = BinLen + buffer_array[buffer_cnt + 5]
                                    if AsciiLen == BinLen:
                                        if buffer_remaining >= AsciiLen + 5:
                                            # if buffer_array[buffer_cnt + AsciiLen + 5] == 0x0A:
                                            #     status = 1  # Valid Message
                                            # else:
                                            #     status = 2  # Unvalid Message Start
                                            status = 1
                                        else:
                                            status = 0  # Uncompleated Message
                                    else:
                                        status = 2  # Unvalid Message Start

                                else:
                                    status = 2  # Unvalid Message Start
                            else:
                                status = 2  # Unvalid Message Start
                        else:
                            status = 2  # Unvalid Message Start
                    else:
                        status = 2  # Unvalid Message Start
                else:
                    status = 0  # Uncompleated Message

                if status == 2:
                    ret = buffer_array[buffer_cnt + 1:buffer_cnt + 1 + buffer_remaining]
                    return ret, status, None

                if status == 0:
                    ret = buffer_array[buffer_cnt:buffer_cnt + buffer_remaining + 1]
                    return ret, status, None

                if status == 1:
                    DataPacket = buffer_array[buffer_cnt:buffer_cnt + AsciiLen + 6]
                    Leftover = buffer_remaining - (AsciiLen + 5)
                    ret = buffer_array[buffer_cnt + AsciiLen + 6:buffer_cnt + AsciiLen + 6 + Leftover]
                    return ret, status, DataPacket

            buffer_cnt += 1

        if status == -1:
            ret = [0]  # No Start Found

    return ret, status, None


def buffer_add(array, bytes_to_add):
    new_array = bytearray(len(array) + len(bytes_to_add))
    if len(array) > 0:
        new_array[:len(array)] = array
    if len(bytes_to_add) > 0:
        new_array[len(array):] = bytes_to_add
    return new_array
