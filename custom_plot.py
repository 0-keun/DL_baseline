import numpy as np
import matplotlib.pyplot as plt

# # 예시용 더미 데이터 생성 (실제 데이터로 교체하세요)
# time = np.linspace(0, 10, 100)  # 시간: 0~10초, 100개 포인트
# Vo = np.sin(time) * 100 + 300   # Vo: [V]
# I_L = np.cos(time) * 5 + 10     # I_L: [A]
# Vin = np.sin(2*time) * 50 + 400  # Vin: [V]

def plot_sub(i, time, Vo, I_L, Vin, axs, x_lim):
    row = i % 3
    col = i // 3
    ax = axs[row, col]

    # 세 개 데이터 그리기
    ax.plot(time, Vo, label='Vo [V]', color='blue', linewidth=1.5)
    ax.plot(time, I_L, label='I_L [A]', color='green', linewidth=1.5)
    ax.plot(time, Vin, label='Vin [V]', color='red', linewidth=1.5)

    ax.set_title(f"Plot {i+1}", fontsize=14)
    ax.grid(ls=":")
    ax.set_xlim((time[0],time[-1]))

    if row == 2:
        ax.set_xlabel("Time [s]", fontsize=12)

    if col == 0:
        ax.set_ylabel("Voltage / Current\n[V], [A]", fontsize=12)

    ax.legend(loc='best', frameon=False, fontsize=10)

def plot_custom(time, Vo, I_L, Vin, Vo_n, I_L_n, Vin_n, Vo_g, I_L_g, Vin_g, s_time, s_Vo, s_I_L, s_Vin, s_Vo_n, s_I_L_n, s_Vin_n, s_Vo_g, s_I_L_g, s_Vin_g):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True, sharex=True)

    plot_sub(0, time, Vo, I_L, Vin, axs, x_lim=(time[0],time[-1]))
    plot_sub(1, time, Vo_n, I_L_n, Vin_n, axs, x_lim=(time[0],time[-1]))
    plot_sub(2, time, Vo_g, I_L_g, Vin_g, axs, x_lim=(time[0],time[-1]))
    plot_sub(3, s_time, s_Vo, s_I_L, s_Vin, axs, x_lim=(s_time[0],s_time[-1]))
    plot_sub(4, s_time, s_Vo_n, s_I_L_n, s_Vin_n, axs, x_lim=(s_time[0],s_time[-1]))
    plot_sub(5, s_time, s_Vo_g, s_I_L_g, s_Vin_g, axs, x_lim=(s_time[0],s_time[-1]))

    plt.show()

    # 저장하려면 아래 라인 추가
    fig.savefig("multi_plot_voltage_current.png", dpi=600, bbox_inches='tight')

# if __name__ == "__main__":
#     # 예시용 더미 데이터 생성 (실제 데이터로 교체하세요)
#     time = np.linspace(0, 10, 100)  # 시간: 0~10초, 100개 포인트
#     Vo = np.sin(time) * 100 + 300   # Vo: [V]
#     I_L = np.cos(time) * 5 + 10     # I_L: [A]
#     Vin = np.sin(2*time) * 50 + 400  # Vin: [V]

#     plot_custom():

import numpy as np
import matplotlib.pyplot as plt

# 각각 subplot으로 따로 그림
def draw_single_subplot(time, Vo, I_L, Vin, title):
    plt.figure(figsize=(15, 7))
    plt.rc('font',size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(time, Vo, label='Real data', color='blue', linewidth=1.5)
    plt.plot(time, I_L, label='Predicted by vanilla model', color='green', linewidth=1.5)
    plt.plot(time, Vin, label='Predicted by proposed model', color='red', linewidth=1.5)
    plt.grid(ls=':')
    # plt.title(title, fontsize=30)
    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("Voltage / Current\n[V], [A]", fontsize=30)
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/'+title, dpi=1200, bbox_inches='tight')

def draw_fault_subplot(time, true, pred, title):
    plt.figure(figsize=(15, 7))
    plt.rc('font',size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(time, true, label='true', color='blue', linewidth=1.5)
    plt.plot(time, pred, label='predicted', color='green', linewidth=1.5)
    plt.grid(ls=':')
    # plt.title(title, fontsize=30)
    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("Signal", fontsize=30)
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/'+title, dpi=1200, bbox_inches='tight')

# 각각 subplot으로 따로 그림
def draw_line_subplot(time, Vo, I_L, Vin, title):
    plt.figure(figsize=(15, 7))
    plt.rc('font',size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(time, Vo, label='Vo [V]', marker='o', linestyle='-',color='blue', linewidth=1.5)
    plt.plot(time, I_L, label='I_L [A]', marker='o', linestyle='-',color='green', linewidth=1.5)
    plt.plot(time, Vin, label='Vin [V]', marker='o', linestyle='-',color='red', linewidth=1.5)
    plt.grid(ls=':')
    plt.title(title, fontsize=30)
    plt.xlabel("Time [s]", fontsize=30)
    plt.ylabel("Voltage / Current\n[V], [A]", fontsize=30)
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/'+title, dpi=1200, bbox_inches='tight')


# 6개 그래프 출력
# draw_single_subplot(0, time, Vo, I_L, Vin, "Plot 1: Original")
# draw_single_subplot(1, time, Vo_n, I_L_n, Vin_n, "Plot 2: Vo_n / I_L_n / Vin_n")
# draw_single_subplot(2, time, Vo_g, I_L_g, Vin_g, "Plot 3: Vo_g / I_L_g / Vin_g")
# draw_single_subplot(3, s_time, s_Vo, s_I_L, s_Vin, "Plot 4: s_Vo / s_I_L / s_Vin")
# draw_single_subplot(4, s_time, s_Vo_n, s_I_L_n, s_Vin_n, "Plot 5: s_Vo_n / s_I_L_n / s_Vin_n")
# draw_single_subplot(5, s_time, s_Vo_g, s_I_L_g, s_Vin_g, "Plot 6: s_Vo_g / s_I_L_g / s_Vin_g")


