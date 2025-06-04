from skins import tile
import copy
class hexagon_map:
    def __init__(self,starting_x, starting_y, a):
        self.hex_row_list=[]
        self.hex_row_list.append(hexagon_row(3, starting_x, starting_y, a, row=0))

        starting_x = starting_x - self.hex_row_list[0].hex_list[0].x_diff
        starting_y = starting_y + a + self.hex_row_list[0].hex_list[0].y_diff
        self.hex_row_list.append(hexagon_row(4, starting_x, starting_y, a, row=1))

        starting_x = starting_x - self.hex_row_list[0].hex_list[0].x_diff
        starting_y = starting_y + a + self.hex_row_list[0].hex_list[0].y_diff
        self.hex_row_list.append(hexagon_row(5, starting_x, starting_y, a, row=2))

        starting_x = starting_x + self.hex_row_list[0].hex_list[0].x_diff
        starting_y = starting_y + a + self.hex_row_list[0].hex_list[0].y_diff
        self.hex_row_list.append(hexagon_row(4, starting_x, starting_y, a, row=3))

        starting_x = starting_x + self.hex_row_list[0].hex_list[0].x_diff
        starting_y = starting_y + a + self.hex_row_list[0].hex_list[0].y_diff
        self.hex_row_list.append(hexagon_row(3, starting_x, starting_y, a, row=4))
        
        self.free_hexes=[]
        self.taken_hexes=[]
        for row in self.hex_row_list:
            for hex in row.hex_list:
                self.free_hexes.append(hex)
                
    def set_mockup_map(self):
        return copy.deepcopy(self)                
                  
class hexagon_row:
    def __init__(self, amount, starting_x, starting_y, a, row):
        self.hex_list=[]

        for i in range(amount):
            if i == 0:
                x = starting_x
                y = starting_y
            else:
                x = self.hex_list[i-1].points[4][0]
                y = self.hex_list[i-1].points[4][1]

            new_hex = hexagon(x, y, a, row, i, tile((255, 255, 255), [], [], 0, None, None, False)) #init hexes as a default tile, row=q, col=r ->convention mistake
            self.hex_list.append(new_hex)

class hexagon: 
    #x+y are coordinatesx, a is a width of hex, q and r are row and column numbers
    def __init__(self, x, y, a, q, r, skin):
        self.q = q
        self.r = r
        self.a = a
        self.skin = skin
        
        self.points = [
            (x, y),
            (x, y + a),
            (x + (3**0.5) * a / 2, y + a * 1.5),
            (x + (3**0.5) * a, y + a),
            (x + (3**0.5) * a, y),
            (x + (3**0.5) * a / 2, y - a / 2),
        ]
        self.x_diff = self.points[2][0] - self.points[1][0] 
        self.y_diff = a/2

    def is_point_inside(self, pos):
        x, y = pos
        n = len(self.points)
        inside = False
        px, py = self.points[-1]
        for cx, cy in self.points:
            if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy) + cx):
                inside = not inside
            px, py = cx, cy
        return inside