import matplotlib.pyplot as plt
import os

# plot the barchart using pyplot for consistent style
# results are just copied from the big results table in report

dir_path = os.path.dirname(os.path.realpath(__file__))

x_labels = ["SIFT", "SIFT + filtering", "SIFT + GMS", "SIFT + GMS\n+ filtering", "ORB", "ORB + filtering", "ORB + GMS", "ORB + GMS\n+ filtering"]
vals = [(0.4+0.2), (0.2+0.4+0.6), (0.2+1), (0.2+1), (0.6+0.6+0.8+0.8+0.4), (0.8+0.4+0.8+0.8+0.4), (0.8+0.8+0.8+1+0.4), (0.8+0.6+0.8+1+0.6)]
heights = [v / 5 * 100 for v in vals]

cmap1 = plt.colormaps["tab20c"]
cmap2 = plt.colormaps["tab20b"]
colours = [cmap1(i) for i in range(4,8)][::-1] + [cmap2(i) for i in range(0,4)][::-1]

plt.figure(figsize = (12,6))
plt.bar(x_labels, heights, width=0.7, color=colours)
plt.title("Average success rate of keypoint algorithms across all objects in test suite")
plt.xlabel("Keypoint algorithm")
plt.ylabel("Average success rate (%)")
plt.ylim(0,85)
plt.savefig(dir_path + "\\barchart.png")
plt.show()