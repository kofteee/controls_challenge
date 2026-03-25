import os
import numpy as np
from . import BaseController


V_LOW  = 15.0
V_HIGH = 25.0
ROLL_THRESHOLD = 1.5

def _e(key, default):
    return float(os.getenv(key, str(default)))


class Controller(BaseController):

    def __init__(self):


        self.p_low   = _e("PID_P_LOW",   0.17438654272285778)
        self.i_low   = _e("PID_I_LOW",   0.10398271419154387)
        self.d_low   = _e("PID_D_LOW",   0.3582928965179535)
        self.f_low   = _e("PID_F_LOW",   0.0334900825201323)

        self.p_mid   = _e("PID_P_MID",   0.25658878539704655)
        self.i_mid   = _e("PID_I_MID",   0.12333965654997325)
        self.d_mid   = _e("PID_D_MID",   0.17169727582735358)
        self.f_mid   = _e("PID_F_MID",   0.05482246813387745)

        self.p_high  = _e("PID_P_HIGH",  0.315911842197468)
        self.i_high  = _e("PID_I_HIGH",  0.11845289388578574)
        self.d_high  = _e("PID_D_HIGH",  0.1819505618452602)
        self.f_high  = _e("PID_F_HIGH",  0.021556928479895595)

        self.w1      = _e("PID_W1",      0.32931227836671695)
        self.w2      = _e("PID_W2",      0.03453168174330681)
        self.w3      = _e("PID_W3",     -0.05879793364099683)
        self.alpha   = _e("PID_ALPHA",   0.6137622062462003)

        self.error_integral = 0.0
        self.prev_error     = 0.0
        self.last_output    = 0.0

    def _get_gains(self, v_ego, roll_lataccel):
        if v_ego < V_LOW:
            p, i, d, f = self.p_low, self.i_low, self.d_low, self.f_low
        elif v_ego >= V_HIGH:
            p, i, d, f = self.p_high, self.i_high, self.d_high, self.f_high
        else:
            mid_v = (V_LOW + V_HIGH) / 2
            if v_ego < mid_v:
                t = (v_ego - V_LOW) / (mid_v - V_LOW)
                p = self.p_low + t * (self.p_mid - self.p_low)
                i = self.i_low + t * (self.i_mid - self.i_low)
                d = self.d_low + t * (self.d_mid - self.d_low)
                f = self.f_low + t * (self.f_mid - self.f_low)
            else:
                t = (v_ego - mid_v) / (V_HIGH - mid_v)
                p = self.p_mid + t * (self.p_high - self.p_mid)
                i = self.i_mid + t * (self.i_high - self.i_mid)
                d = self.d_mid + t * (self.d_high - self.d_mid)
                f = self.f_mid + t * (self.f_high - self.f_mid)

        if abs(roll_lataccel) > ROLL_THRESHOLD:
            i = i * (ROLL_THRESHOLD / max(abs(roll_lataccel), 1e-6))

        return p, i, d, f

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        p, i, d, f = self._get_gains(state.v_ego, state.roll_lataccel)

        self.error_integral += error
        self.error_integral  = max(min(self.error_integral, 5.0), -5.0)

        error_diff      = error - self.prev_error
        self.prev_error = error

        ff = f * target_lataccel
        if future_plan and len(future_plan.lataccel) >= 15:
            ff += self.w1 * future_plan.lataccel[4]
            ff += self.w2 * future_plan.lataccel[9]
            ff += self.w3 * future_plan.lataccel[14]

        raw    = (p * error) + (i * self.error_integral) + (d * error_diff) + ff
        smooth = self.alpha * self.last_output + (1.0 - self.alpha) * raw
        self.last_output = smooth
        return smooth