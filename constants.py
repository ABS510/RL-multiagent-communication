SMALL_SHAPE = (4, 8)
LARGE_SHAPE = (4, 16)
BALL_SHAPE = (4, 2)
BORDER_CORNER_SHAPE = (10, 8)
BOARD_TOP = 24
VIDEO_FRAMES = 500

large_trimmed_rows = (96, 146)
small_trimmed_rows = (64, 114)
search_trimmed_rows = [*large_trimmed_rows, *small_trimmed_rows]

LARGE_ROWS = [BOARD_TOP + r for r in large_trimmed_rows]
SMALL_ROWS = [BOARD_TOP + r for r in small_trimmed_rows]
SEARCH_ROWS = [BOARD_TOP + r for r in search_trimmed_rows]
