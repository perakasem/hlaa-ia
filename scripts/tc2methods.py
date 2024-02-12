def rk2(tc, y, v, h, b, k, v_prev):
    k1_v = tc.function(y, v, b, k)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y
    k2_v = tc.function(midpoint_y, midpoint_v, b, k)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v
    return y_next, v_next, None

def euler(tc, y, v, h, b, k, v_prev):
    v_next = v + h * tc.function(y, v, b, k)
    y_next = y + h * v
    return y_next, v_next, None

def pc(tc, y, v, h, b, k, v_prev):
    # Predictor (Euler method)
    v_predict = v + h * tc.function(y, v, b, k)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * tc.function(y_predict, v_predict, b, k) - 0.5 * tc.function(y, v_prev, b, k))
    else:
        # For the first step, fall back to Euler method as we don't have v_prev
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict