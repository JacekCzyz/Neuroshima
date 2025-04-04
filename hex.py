import numpy as np
import copy
from skins import tile

class hexagon_map:
    def __init__(self, starting_x, starting_y, a):
        self.hex_row_list = []
        self.hex_row_list.append(hexagon_row(3, starting_x, starting_y, a, row=0))

        x_diff = self.hex_row_list[0].hex_list[0].x_diff
        y_diff = self.hex_row_list[0].hex_list[0].y_diff

        starting_x = starting_x - x_diff
        starting_y = starting_y + a + y_diff
        self.hex_row_list.append(hexagon_row(4, starting_x, starting_y, a, row=1))

        starting_x = starting_x - x_diff
        starting_y = starting_y + a + y_diff
        self.hex_row_list.append(hexagon_row(5, starting_x, starting_y, a, row=2))

        starting_x = starting_x + x_diff
        starting_y = starting_y + a + y_diff
        self.hex_row_list.append(hexagon_row(4, starting_x, starting_y, a, row=3))

        starting_x = starting_x + x_diff
        starting_y = starting_y + a + y_diff
        self.hex_row_list.append(hexagon_row(3, starting_x, starting_y, a, row=4))

        self.free_hexes = []
        self.taken_hexes = []

        for row in self.hex_row_list:
            self.free_hexes.extend(row.hex_list)

    def set_mockup_map(self):
        return copy.deepcopy(self)

class hexagon_row:
    def __init__(self, amount, starting_x, starting_y, a, row):
        self.hex_list = []
        for i in range(amount):
            if i == 0:
                x, y = starting_x, starting_y
            else:
                x = self.hex_list[i-1].points[4][0]
                y = self.hex_list[i-1].points[4][1]

            default_skin = tile()
            new_hex = hexagon(x, y, a, row, i, default_skin)
            self.hex_list.append(new_hex)

        # Convert hex_list to a NumPy array
        self.hex_list = np.array(self.hex_list, dtype=object)



class hexagon:
    def __init__(self, x, y, a, q, r, skin):
        self.q = q
        self.r = r
        self.a = a
        self.skin = skin

        sqrt3 = np.sqrt(3)
        self.points = np.array([
            [x, y],
            [x, y + a],
            [x + sqrt3 * a / 2, y + 1.5 * a],
            [x + sqrt3 * a, y + a],
            [x + sqrt3 * a, y],
            [x + sqrt3 * a / 2, y - a / 2],
        ])

        self.x_diff = self.points[2, 0] - self.points[1, 0]
        self.y_diff = a / 2

    def is_point_inside(self, pos):
        x, y = pos
        n = self.points.shape[0]
        inside = False
        px, py = self.points[-1]

        for cx, cy in self.points:
            if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy) + cx):
                inside = not inside
            px, py = cx, cy

        return inside
