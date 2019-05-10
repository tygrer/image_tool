import numpy as np
import math


sum_val = 0
def loss(fc_out, label):
    for i, la in enumerate(label):
            val = 0
            f_val = 0

            for j, fv in enumerate(fc_out[i]):
                val = val + i*math.log(fc_out[i,j])
                f_val = f_val + i * math.log(1-fc_out[i, j])
            sum_val = sum_val -val*la - f_val*(1-la)

    return sum_val
if __name__ == "__main__":
    label = np.array([0, 0, 1])
    fc_out = np.array([
        [2.5, -2, 0.8989],
        [3, 0.8, -842],
        [0.00000000000001, 2, 4.9]
    ])
    val = loss(fc_out,label)
    print(val)