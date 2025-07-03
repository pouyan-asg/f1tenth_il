def render_callback(env_renderer):
    """
    This function is a custom rendering helper for a car racing environment. 
    It adjusts the camera and UI elements to follow the car during rendering.

    - Extracts the car's vertices (positions).
    - Calculates the bounding box (top, bottom, left, right) of the car.
    - Moves the score label to just above the car.
    - Adjusts the camera view (e.left, e.right, e.top, e.bottom) so 
        the car stays centered, with a margin of 800 units on each side.

    Note:
    standard Gym or Gymnasium environments do not require an add_render_callback method;
    this is not part of the official Gym API.
    This environment (f110-v0) is a custom environment for autonomous racing (F1TENTH).
    """

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800