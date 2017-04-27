import rospy
from geometry_msgs.msg import Transform


class CSVData:
    def __init__(self):
        self.rows = []
        self.tsec_rows = dict()

    def add_row(self, row):
        self.rows.append(row)
        if row.tsec in self.tsec_rows.keys():
            self.tsec_rows[row.tsec].append(len(self.rows) - 1)
            # keep the list of nanoseconds sorted (every second has a sorted list of nanoseconds)
            self.tsec_rows[row.tsec].sort(key=lambda idx: self.rows[idx].tnsec)
        else:
            self.tsec_rows[row.tsec] = [len(self.rows) - 1]

    def find_closest_row(self, tsec, tnsec):
        try:
            lower_idx = self.find_closest_nsec(self.tsec_rows[tsec], tnsec)
        except KeyError:
            print "sec not found"

    def find_closest_nsec(self, nsec_row_idx_arr, tnsec):
        min = 0
        max = len(nsec_row_idx_arr) - 1
        # binary search for closest nsec
        while True:
            if max < min:
                return -1
            idx = (max + min) // 2
            if self.rows[nsec_row_idx_arr[idx]].tnsec < tnsec:
                min = idx + 1
            elif self.rows[nsec_row_idx_arr[idx]].tnsec > tnsec:
                max = idx - 1
            else:
                return idx



class CSVRow:
    def __init__(self, tsec, tnsec, image_id):
        self.tsec = tsec
        self.tnsec = tnsec
        self.image_id = image_id
        self.tf = []
        self.force = []


class TF:
    def __init__(self, tf_data):
        self.set_data(tf_data[0], tf_data[1], float(tf_data[2]), float(tf_data[3]), float(tf_data[4]),
                      float(tf_data[5]), float(tf_data[6]), float(tf_data[7]), float(tf_data[8]))

    def set_data(self, frame_id, child_frame_id, x, y, z, qw, qx, qy, qz):
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.transform = Transform()
        self.transform.translation.x = x
        self.transform.translation.y = y
        self.transform.translation.z = z
        self.transform.rotation.w = qw
        self.transform.rotation.x = qx
        self.transform.rotation.y = qy
        self.transform.rotation.z = qz


def load_csv(csv, fname):
    csv_data = CSVData()
    for i in range(len(csv)):
        row = load_row(csv[i])
        csv_data.add_row(row)

    return csv_data


def load_row(raw_row):
    tsec = int(raw_row[0])
    tnsec = int(raw_row[1])
    image_id = int(raw_row[2])
    row = CSVRow(tsec, tnsec, image_id)

    i = 3
    while i < raw_row.size:
        entry_type = check_entry(raw_row[i])
        if entry_type == 'tf':
            row.tf.append(TF(raw_row[i:i+9]))
            i += 9
        elif entry_type == 'force':
            row.force = [float(val) for val in raw_row[i:i+26]]
            break # finished after force

    return row


def check_entry(cell):
    # check if cell is a tf
    if cell[0].isalpha() or cell[0] == '/':
        return 'tf'
    else:
        return 'force'