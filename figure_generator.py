import matplotlib.pyplot as plt
import matplotlib

fig,ax = plt.subplots()

rect1 = matplotlib.patches.Rectangle((0.00,0.9),0.2,0.03,color="#36cf1f")
rect2 = matplotlib.patches.Rectangle((0.00,0.95),0.2,0.03,color="#337733")

rect3 = matplotlib.patches.Rectangle((0.40,0.9),0.2,0.03,color="#1f3fcf")
rect4 = matplotlib.patches.Rectangle((0.40,0.95),0.2,0.03,color="#396296")

rect5 = matplotlib.patches.Rectangle((0.80,0.9),0.2,0.03,color="#337733")
rect6 = matplotlib.patches.Rectangle((0.80,0.95),0.2,0.03,color="#337733")




ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
#ax.set_axis_off()

plt.show()

