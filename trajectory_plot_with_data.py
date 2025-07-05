import csv
from scipy import interpolate
import numpy as np
    
import matplotlib.ticker as ticker

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
     


# ========= REFERENCE TRAJECTORY DATA =================
cx = []
cy = []
cz = []
cyaw = []
cpitch = []
croll = []

csv_filename = 'gazebo_reference_trajectory_data.csv'


with open(csv_filename, mode='r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        cx.append(float(row[0]))
        cy.append(float(row[1]))
        cz.append(float(row[2]))
        cyaw.append(float(row[3]))
        cpitch.append(float(row[4]))
        croll.append(float(row[5]))
        
        
cx = np.array(cx)
cy = np.array(cy)
cz = np.array(cz)
cyaw = np.array(cyaw)
cpitch = np.array(cpitch)
croll = np.array(croll)




# ========= transformed jacobian COUPLED TRAJECTORY DATA =================
tx_coup = []
ty_coup = []
tz_coup = []
tyaw_coup = []
tpitch_coup = []
troll_coup = []
csv_filename = 'coupled_tracking_trajectory_data.csv'

with open(csv_filename, mode='r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        # x_coup.append(float(row[0][1:-1]))
        tx_coup.append(float(row[0]))
        ty_coup.append(float(row[1]))
        # y_coup.append(float(row[1][2:-2]))
        tz_coup.append(float(row[2]))
        tyaw_coup.append(float(row[3]))
        tpitch_coup.append(float(row[4]))
        troll_coup.append(float(row[5]))
# print(z_coup[0])        
# tx_coup[0] = cx[0]
# ty_coup[0] = cy[0]
# tz_coup[0] = cz[0]
# tyaw_coup[0] = cyaw[0]
# tpitch_coup[0] = cpitch[0]
# troll_coup[0] = croll[0]
# print(cx[:144])
tx = np.array(tx_coup)
ty = np.array(ty_coup)
tz = np.array(tz_coup)
tyaw = np.array(tyaw_coup)
tpitch = np.array(tpitch_coup)
troll = np.array(troll_coup)
   
  






# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # ax.plot(x_coup,y_coup, z_coup, "-b", label="Coupled Tracking")


# ax.plot(cx[:500],cy[:500],cz[:500],  "-b", linewidth=1, label="Reference Trajectory with Human Uncertainty")
# # ax.plot(wn_cx[:291],wn_cy[:291],wn_cz[:291],  "-g", linewidth=0.5, label="wn_Reference Trajectory")
# ax.plot(tx_coup,ty_coup, tz_coup, "-r",linewidth=2, label="Whole-Body Tracking NPO-HU")
# # ax.plot(tx_coup1,ty_coup1, tz_coup1, "-m",linewidth=2, label="Whole-Body Tracking PO-NHU")
# # ax.plot(x_opt,y_opt, z_opt, "--g",linewidth=2, label="Whole-Body Tracking PO-HU")

# ax.grid(True)
# ax.set_xlabel("x", fontsize=30)
# ax.set_ylabel("y", fontsize=30)
# ax.set_zlabel("z", fontsize=30)
# # ax.set_title("Coupled Tracking and Pose Optimized Tracking", fontsize=25, fontweight="bold")
# ax.legend(fontsize=24)
# ax.tick_params(axis='both', labelsize=30)
# ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))



# # PLOT 3D POINTS

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # ax.plot(x_coup, y_coup, z_coup, "-b", label="Coupled Tracking")

# ax.plot(cx[:500], cy[:500], cz[:500], "ob", markersize=4, label="Reference Trajectory with HU")
# # ax.plot(wn_cx[:291], wn_cy[:291], wn_cz[:291], "og", markersize=2, label="wn_Reference Trajectory")
# ax.plot(tx_coup, ty_coup, tz_coup, "or", markersize=4, label="Tracking: NPO-HU")
# # ax.plot(tx_coup1, ty_coup1, tz_coup1, "om", markersize=4, label="Tracking: PO-NHU")
# # ax.plot(x_opt, y_opt, z_opt, "og", markersize=4, label="Tracking: PO-HU")

# ax.grid(True)
# ax.set_xlabel("x", fontsize=30)
# ax.set_ylabel("y", fontsize=30)
# ax.set_zlabel("z", fontsize=30)
# # ax.set_title("Coupled Tracking and Pose Optimized Tracking", fontsize=25, fontweight="bold")
# ax.legend(fontsize=24)
# ax.tick_params(axis='both', labelsize=30)
# ax.zaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))


# plt.show()










# JUST PLOT THE POINTS.

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot x-axis tracking
axs[0].plot(cx[:991], "ob", markersize=2, label="Nominal Trajectory with HU")
# axs[0].plot(tx_coup1, "om", markersize=2, label="Whole-Body Tracking PO-NHU")
axs[0].plot(tx_coup, "or", markersize=2, label="Whole-Body Tracking NPO-HU")
# axs[0].plot(x_opt, "og", markersize=2, label="Whole-Body Tracking PO-HU")

axs[0].grid(True)
axs[0].set_ylabel("x-axis", fontsize=22, fontname="Times New Roman")
axs[0].legend(fontsize=15)

# Plot y-axis tracking
axs[1].plot(cy[:991], "ob", markersize=2)
# axs[1].plot(ty_coup1, "om", markersize=2)
axs[1].plot(ty_coup, "or", markersize=2)
# axs[1].plot(y_opt, "og", markersize=2)

axs[1].grid(True)
axs[1].set_ylabel("y-axis", fontsize=22, fontname="Times New Roman")

# Plot z-axis tracking
axs[2].plot(cz[:991], "ob", markersize=2)
# axs[2].plot(tz_coup1, "om", markersize=2)
axs[2].plot(tz_coup, "or", markersize=2)
# axs[2].plot(z_opt, "og", markersize=2)

axs[2].grid(True)
axs[2].set_xlabel("Time Step", fontsize=22, fontname="Times New Roman")
axs[2].set_ylabel("z-axis", fontsize=22, fontname="Times New Roman")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


