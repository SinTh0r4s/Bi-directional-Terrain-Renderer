def map(index: int, num_columns: int) -> tuple[int, int]:
    vertices_per_row = num_columns * 2 # bitshift
    square_row = int(index / vertices_per_row)
    in_square_row_id = index % vertices_per_row
    row_offset = in_square_row_id % 2 # & 0x1
    row = square_row + row_offset
    col = int(in_square_row_id / 2)
    if square_row % 2:
        col = num_columns - 1 - col
    return row, col


for x in range(18):
    row, col = map(x, 3)
    print(f"{x} -> {row}; {col}")