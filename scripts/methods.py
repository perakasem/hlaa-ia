def rk2(tc, y, v, h, omega, v_prev):
    k1_v = tc.function(y, omega)
    k1_y = v
    midpoint_v = v + 0.5 * h * k1_v
    midpoint_y = y + 0.5 * h * k1_y
    k2_v = tc.function(midpoint_y, omega)
    k2_y = midpoint_v
    y_next = y + h * k2_y
    v_next = v + h * k2_v
    return y_next, v_next, None

def euler(tc, y, v, h, omega, v_prev):
    v_next = v + h * tc.function(y, omega)
    y_next = y + h * v
    return y_next, v_next, None

def pc(tc, y, v, h, omega, v_prev):
    # Predictor (Euler method)
    v_predict = v + h * tc.function(y, omega)
    y_predict = y + h * v

    # Corrector (Adams-Bashforth method)
    if v_prev is not None:
        v_correct = v + h * (1.5 * tc.function(y_predict, omega) - 0.5 * tc.function(y, omega))
    else:
        v_correct = v_predict
    y_correct = y + 0.5 * h * (v + v_correct)

    return y_correct, v_correct, v_predict