import numpy as np


def is_in_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


def blocks_light(grid, x, y):
    return not is_in_bounds(x, y, grid.shape[1], grid.shape[0]) or grid[y][x] == 1


def set_visible(visible, x, y):
    visible[y][x] = True


def cast_light(grid, visible, cx, cy, row, start, end, radius, xx, xy, yx, yy):
    if start < end:
        return
    radius_squared = radius * radius
    for i in range(row, radius + 1):
        dx, dy = -i - 1, -i
        blocked = False
        new_start = start
        while dx <= 0:
            dx += 1
            X = cx + dx * xx + dy * xy
            Y = cy + dx * yx + dy * yy
            l_slope = (dx - 0.5) / (dy + 0.5)
            r_slope = (dx + 0.5) / (dy - 0.5)
            if start < r_slope:
                continue
            elif end > l_slope:
                break
            else:
                if dx * dx + dy * dy < radius_squared:
                    if is_in_bounds(X, Y, grid.shape[1], grid.shape[0]):
                        set_visible(visible, X, Y)
                if blocked:
                    if blocks_light(grid, X, Y):
                        new_start = r_slope
                        continue
                    else:
                        blocked = False
                        start = new_start
                else:
                    if blocks_light(grid, X, Y) and i < radius:
                        blocked = True
                        cast_light(grid, visible, cx, cy, i + 1, start, l_slope, radius, xx, xy, yx, yy)
                        new_start = r_slope
        if blocked:
            break


def compute_fov(grid, x, y, radius):
    height, width = grid.shape
    visible = np.zeros_like(grid, dtype=bool)
    set_visible(visible, x, y)  # 起点可见

    # 八个象限
    for xx, xy, yx, yy in [(1, 0, 0, 1), (0, 1, 1, 0), (-1, 0, 0, 1), (0, -1, 1, 0),
                           (1, 0, 0, -1), (0, 1, -1, 0), (-1, 0, 0, -1), (0, -1, -1, 0)]:
        cast_light(grid, visible, x, y, 1, 1.0, 0.0, radius, xx, xy, yx, yy)
    return visible


if __name__ == "__main__":
    # 创建一个简单的地图，0 表示透明，1 表示墙体
    grid = np.zeros((10, 10), dtype=int)
    grid[4][4:7] = 1  # 墙体阻挡

    # 起点和视距
    start_x, start_y = 5, 5
    radius = 5

    visible = compute_fov(grid, start_x, start_y, radius)

    # 可视化结果
    for y in range(grid.shape[0]):
        row = ""
        for x in range(grid.shape[1]):
            if x == start_x and y == start_y:
                row += "O"
            elif grid[y][x] == 1:
                row += "#"
            elif visible[y][x]:
                row += "."
            else:
                row += " "
        print(row)
