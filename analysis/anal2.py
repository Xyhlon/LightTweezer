import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import linear_model
from scipy.constants import k as k
import numpy.ma as ma
from uncertainties import ufloat


# Load data
filename = "I2.avi"
output = filename.split(".")[0]+"_tracked_obj.npy"
# output = "temp.npy"
t_obj = np.load(output)

video = cv2.VideoCapture(filename)
ret,frame = video.read()
fps = video.get(cv2.CAP_PROP_FPS)
delta_t = 1/fps

# Set thresholds and constants
# variance_threshold = 0.3 for trapped
variance_threshold = 20
mupx = 130e-6 / 1440  # micrometers per pixel (X-axis)
mupy = 98e-6 / 1080   # micrometers per pixel (Y-axis)
aspect_ratio = 1

frames = t_obj.shape[2]
p_non_zero = np.count_nonzero(t_obj, axis=2)/frames
mask = np.all(p_non_zero > 0.5, axis=1)
t_obj = t_obj[mask]


mask = t_obj == 0
t_obj = ma.masked_array(t_obj, mask=mask)
# Calculate variances and apply a mask
variances = np.var(t_obj, axis=2)
mask = np.all(variances > variance_threshold, axis=1)
t_obj = t_obj[mask]

# print(t_obj)

# print(variances)

# Update plot settings
rcsettings = {
    "font.size": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern} \usepackage[locale=DE, uncertainty-mode=separate]{siunitx} \DeclareSIUnit{\px}{px}",
    "font.family": "Latin Modern Roman",
    "legend.fontsize": 10,
}
plt.rcParams.update(rcsettings)

# Create the plot
fig, ax = plt.subplots()

for (i_,part) in enumerate(t_obj):
    i = i_ + 1
    part = ma.filled(part, 0)
    mask = np.all(part != 0 , axis=0)
    part = part[:,mask]
    # Rescaling displacement and position
    scale = np.asarray([mupx, mupy]).reshape(-1, 1)
    original_pos = part[:, 0].reshape(-1, 1)
    rescaled_disp = scale * (part - original_pos)
    rescaled_pos = scale * part
    x = part[0]
    y = part[1]
    
    # Preparing data for gradient line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    num_segments = len(segments)
    
    # Create a LineCollection
    lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(0, num_segments))
    lc.set_array(np.arange(num_segments))  # Color the segments by their order
    lc.set_linewidth(2)  # Set line width
    lc.set_alpha(np.linspace(0.1, 1, num_segments))  # Alpha gradient from start to end of the line
    ax.add_collection(lc)
    
    # Add label to origin of the particle
    ax.text(x[0]+10, y[0]-15, f"  {i}", fontsize=11, color='black', weight='bold')
    
    mean_sqrd_displacement_time_series = np.cumsum((rescaled_disp)**2, axis=1)/np.arange(1, rescaled_disp.shape[1]+1)

    fig_msd, ax_msd = plt.subplots()
    ax_msd.plot(mean_sqrd_displacement_time_series[0], label="X")
    ax_msd.plot(mean_sqrd_displacement_time_series[1], label="Y")
    ax_msd.plot(mean_sqrd_displacement_time_series[0]+mean_sqrd_displacement_time_series[1], label="R")
    time_ax = ax_msd.secondary_xaxis('top', functions=(lambda x: x * delta_t, lambda x: x / delta_t))
    ax_msd.set_xlabel(r"$n$")
    time_ax.set_xlabel(r"$t_n$ / \si{\second}")
    ax_msd.set_ylabel(r"$\left\langle \Delta s^2 \right\rangle(t_n)$ / \si{\meter\squared}")
    ax_msd.set_title(f"Mean squared displacement over time for particle {i}")

    ransac = linear_model.RANSACRegressor(max_trials=5000,loss="squared_error", residual_threshold=1)
    y = (mean_sqrd_displacement_time_series[0]+mean_sqrd_displacement_time_series[1]).reshape(-1, 1)
    x = (np.arange(1, len(y)+1)).reshape(-1, 1)
    ransac.fit(x, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    lw = 2
    color = "purple"
    ax_msd.plot(
        line_X,
        line_y_ransac,
        color=color,
        linewidth=lw,
        label="RANSAC regressor",
    )
    slope = ransac.estimator_.coef_[0][0]
    max_idx = np.argmax(y[inlier_mask] - ransac.predict(x[inlier_mask]))
    min_idx =  np.argmin(y[inlier_mask] - ransac.predict(x[inlier_mask]))
    y_max = y[inlier_mask][max_idx]
    y_min = y[inlier_mask][min_idx]
    d_max = y_max - slope * max_idx
    d_min = y_min - slope * min_idx
    line_upper = slope * line_X + d_max
    line_lower = slope * line_X + d_min
    ax_msd.plot(
        line_X,
        line_upper,
        color=color,
        linewidth=lw,
        linestyle="--",
        label="RANSAC upper bound",
    )
    ax_msd.plot(
            line_X,
            line_lower,
        color=color,
        linewidth=lw,
        linestyle="--",
        label="RANSAC lower bound",
    )
    slope_si = slope / delta_t
    delta_y = y[1:]-y[:-1] 
    max_dy = np.max(delta_y)
    slope_si = max_dy / delta_t
    R = 2.06e-6/2
    T = ufloat(308.15,15)
    eta = 2* k*T / (3*np.pi*R*slope_si)
    text = r" $\frac{\partial\left\langle \Delta s^2 \right\rangle(t_n)}{\partial t}$"+ f" = {slope_si:.2e} " + r"\si{\meter\squared\per\second}"
    text = text + "\n" + r"$\eta_\text{eff}$ = " + f"{eta:.2e} " + r"\si{\pascal\second}"
    print(eta*1e3)



    xmin, xmax = ax_msd.get_xlim()
    ymin, ymax = ax_msd.get_ylim()
    offset = (55, -10)
    # discuss that the heat of the lamp can easily change the viscosity


    ax_msd.text(s=text,
                bbox={"facecolor": color, "alpha": 0.85},
                x=(xmax - xmin) / 100 * (5 + offset[0]) + xmin,
                y=(ymax - ymin) * (offset[1] / 100 + 1 / 7) + ymin,
                fontsize=10,
            )

    ax_msd.legend()
    plt.savefig(f"{filename.split(".")[0]}_particle_{i}.pdf")


# Axis and title setup
ax.set_xlim(0, 1440)
ax.set_ylim(0, 1080)
ax.invert_yaxis()
ax.set_aspect(aspect_ratio)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title("Tracked Brownian motion of particles")

# Set background to image
ax.imshow(frame, zorder=0)

# Adding secondary axis for real space scale
def px_to_micron_x(x):
    return x * mupx * 1e6  # Convert to micrometers

def px_to_micron_y(y):
    return y * mupy * 1e6  # Convert to micrometers

secax_x = ax.secondary_xaxis('top', functions=(px_to_micron_x, lambda x: x / (mupx * 1e6)))
secax_y = ax.secondary_yaxis('right', functions=(px_to_micron_y, lambda y: y / (mupy * 1e6)))
secax_x.set_xlabel("x (micrometers)")
secax_y.set_ylabel("y (micrometers)")

# fig.savefig(f"{filename.split('.')[0]}_tracked.pdf")

# plt.show()

