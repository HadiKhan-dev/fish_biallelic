import matplotlib.pyplot as plt
import matplotlib

fig,ax = plt.subplots()
fig.set_size_inches((18,12))

ax.set_xlim((-0.2,1))
ax.set_ylim((0,1))

rect1 = matplotlib.patches.Rectangle((0.00,0.95),0.2,0.03,color="#337733")
rect2 = matplotlib.patches.Rectangle((0.00,0.90),0.2,0.03,color="#36cf1f")

rect3 = matplotlib.patches.Rectangle((0.40,0.95),0.2,0.03,color="#396296")
rect4 = matplotlib.patches.Rectangle((0.40,0.90),0.2,0.03,color="#1f3fcf")

rect5 = matplotlib.patches.Rectangle((0.80,0.95),0.2,0.03,color="#7333bd")
rect6 = matplotlib.patches.Rectangle((0.80,0.90),0.2,0.03,color="#ab05e8")


ax.arrow(0.12,0.85,0.15,-0.2,width=0.005,color="k")
ax.arrow(0.48,0.85,-0.15,-0.2,width=0.005,color="k")
ax.arrow(0.52,0.85,0.15,-0.2,width=0.005,color="k")
ax.arrow(0.88,0.85,-0.15,-0.2,width=0.005,color="k")

rect7 = matplotlib.patches.Rectangle((0.20,0.57),0.1,0.03,color="#337733")
rect8 = matplotlib.patches.Rectangle((0.30,0.57),0.1,0.03,color="#36cf1f")

rect9 = matplotlib.patches.Rectangle((0.20,0.52),0.15,0.03,color="#396296")
rect10 = matplotlib.patches.Rectangle((0.35,0.52),0.05,0.03,color="#1f3fcf")

rect11 = matplotlib.patches.Rectangle((0.60,0.57),0.05,0.03,color="#7333bd")
rect12 = matplotlib.patches.Rectangle((0.65,0.57),0.15,0.03,color="#ab05e8")

rect13 = matplotlib.patches.Rectangle((0.60,0.52),0.12,0.03,color="#1f3fcf")
rect14 = matplotlib.patches.Rectangle((0.72,0.52),0.08,0.03,color="#396296")

ax.arrow(0.32,0.45,0.15,-0.2,width=0.005,color="k")
ax.arrow(0.68,0.45,-0.15,-0.2,width=0.005,color="k")

rect15 = matplotlib.patches.Rectangle((0.40,0.17),0.07,0.03,color="#337733")
rect16 = matplotlib.patches.Rectangle((0.47,0.17),0.08,0.03,color="#396296")
rect17 = matplotlib.patches.Rectangle((0.55,0.17),0.05,0.03,color="#1f3fcf")


rect18 = matplotlib.patches.Rectangle((0.40,0.12),0.05,0.03,color="#7333bd")
rect19 = matplotlib.patches.Rectangle((0.45,0.12),0.05,0.03,color="#ab05e8")
rect20 = matplotlib.patches.Rectangle((0.50,0.12),0.02,0.03,color="#1f3fcf")
rect21 = matplotlib.patches.Rectangle((0.52,0.12),0.08,0.03,color="#396296")



ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
ax.add_patch(rect8)
ax.add_patch(rect9)
ax.add_patch(rect10)
ax.add_patch(rect11)
ax.add_patch(rect12)
ax.add_patch(rect13)
ax.add_patch(rect14)
ax.add_patch(rect15)
ax.add_patch(rect16)
ax.add_patch(rect17)
ax.add_patch(rect18)
ax.add_patch(rect19)
ax.add_patch(rect20)
ax.add_patch(rect21)

ax.text(-0.12,0.92,s="P",fontsize=40)
ax.text(-0.12,0.535,s="F1",fontsize=40)
ax.text(-0.12,0.135,s="F2",fontsize=40)

ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,8))

ax.set_xlim((0,1))
ax.set_ylim((0.2,0.7))

rect1 = matplotlib.patches.Rectangle((0.10,0.45),0.03,0.03,color="#337733")
rect2 = matplotlib.patches.Rectangle((0.13,0.45),0.08,0.03,color="#396296")
rect3 = matplotlib.patches.Rectangle((0.21,0.45),0.06,0.03,color="#ab05e8")
rect4 = matplotlib.patches.Rectangle((0.27,0.45),0.03,0.03,color="#36cf1f")

rect5 = matplotlib.patches.Rectangle((0.10,0.40),0.12,0.03,color="#36cf1f")
rect6 = matplotlib.patches.Rectangle((0.22,0.40),0.03,0.03,color="#7333bd")
rect7 = matplotlib.patches.Rectangle((0.25,0.40),0.05,0.03,color="#396296")

rect8 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
rect9 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
rect10 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")

rect11 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
rect12 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

rect13 = matplotlib.patches.Rectangle((0.61,0.48),0.05,0.03,color="#36cf1f")
rect14 = matplotlib.patches.Rectangle((0.66,0.48),0.11,0.03,color="#7333bd")
rect15 = matplotlib.patches.Rectangle((0.77,0.48),0.04,0.03,color="#ab05e8")

rect16 = matplotlib.patches.Rectangle((0.61,0.43),0.1,0.03,color="#1f3fcf")
rect17 = matplotlib.patches.Rectangle((0.71,0.43),0.02,0.03,color="#337733")
rect18 = matplotlib.patches.Rectangle((0.73,0.43),0.08,0.03,color="#36cf1f")

oval1 = matplotlib.patches.Ellipse((0.45,0.45),0.8,0.4,color="r",fill=False,linewidth=10)

ax.text(0.87,0.43,s="x290",fontsize=50)

#rect7 = matplotlib.patches.Rectangle((0.67,0.43),0.2,0.03,color="#ab05e8")


#rect8 = matplotlib.patches.Rectangle((0.57,0.51),0.2,0.03,color="#7333bd")
#rect9 = matplotlib.patches.Rectangle((0.64,0.47),0.2,0.03,color="#ab05e8")

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
ax.add_patch(rect8)
ax.add_patch(rect9)
ax.add_patch(rect10)
ax.add_patch(rect11)
ax.add_patch(rect12)
ax.add_patch(rect13)
ax.add_patch(rect14)
ax.add_patch(rect15)
ax.add_patch(rect16)
ax.add_patch(rect17)
ax.add_patch(rect18)

ax.add_patch(oval1)

ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,15))

ax.set_xlim((0.33,0.72))
ax.set_ylim((0.25,0.45))

#ax.set_xlim((0,1))
#ax.set_ylim((0,1))


rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
rect2 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")


rect4 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
rect5 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

oval1 = matplotlib.patches.Ellipse((0.53,0.345),0.1,0.06,color="r",fill=False,linewidth=10)

ax.arrow(0.57,0.32,0.01,-0.01,width=0.003,color="k")

rect6 = matplotlib.patches.Rectangle((0.595,0.27),0.094,0.03,color="k",fill=False,linewidth=4)

ax.text(0.56,0.284,s="...",fontsize=70)

ax.text(0.6,0.279,s="0",fontsize=70)
ax.plot((0.62,0.62),(0.27,0.3),color="k",linewidth=3)

ax.text(0.623,0.279,s="1",fontsize=70)
ax.plot((0.643,0.643),(0.27,0.3),color="k",linewidth=3)

ax.text(0.647,0.279,s="1",fontsize=70)
ax.plot((0.666,0.666),(0.27,0.3),color="k",linewidth=3)

ax.text(0.669,0.279,s="0",fontsize=70)

ax.text(0.695,0.284,s="...",fontsize=70)


ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(oval1)


ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,18))

ax.set_xlim((0.33,0.62))
ax.set_ylim((0.25,0.44))

#ax.set_xlim((0,1))
#ax.set_ylim((0,1))


rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
rect2 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")


rect4 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
rect5 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

oval1 = matplotlib.patches.Ellipse((0.53,0.37),0.1,0.1,color="r",fill=False,linewidth=10)

ax.text(0.498,0.365,s="+",fontsize=70)
ax.text(0.518,0.365,s="+",fontsize=70)
ax.text(0.538,0.365,s="+",fontsize=70)

ax.arrow(0.53,0.318,0.0,-0.008,width=0.003,color="k")

rect6 = matplotlib.patches.Rectangle((0.485,0.265),0.094,0.03,color="k",fill=False,linewidth=4)

ax.text(0.455,0.28,s="...",fontsize=70)

ax.text(0.490,0.2755,s="1",fontsize=70)
ax.plot((0.508,0.508),(0.265,0.295),color="k",linewidth=3)

ax.text(0.513,0.2755,s="1",fontsize=70)
ax.plot((0.531,0.531),(0.265,0.295),color="k",linewidth=3)

ax.text(0.5360,0.2755,s="2",fontsize=70)
ax.plot((0.555,0.555),(0.265,0.295),color="k",linewidth=3)

ax.text(0.560,0.2755,s="0",fontsize=70)

ax.text(0.588,0.28,s="...",fontsize=70)


ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(oval1)


ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,18))

ax.set_xlim((0.33,0.62))
ax.set_ylim((0.25,0.44))

#ax.set_xlim((0,1))
#ax.set_ylim((0,1))


rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
rect2 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")


rect4 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
rect5 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

oval1 = matplotlib.patches.Ellipse((0.53,0.37),0.1,0.1,color="r",fill=False,linewidth=10)

ax.text(0.498,0.365,s="+",fontsize=70)
ax.text(0.518,0.365,s="+",fontsize=70)
ax.text(0.538,0.365,s="+",fontsize=70)

ax.arrow(0.53,0.318,0.0,-0.008,width=0.003,color="k")

rect6 = matplotlib.patches.Rectangle((0.485,0.265),0.094,0.03,color="k",fill=False,linewidth=4)

ax.plot((0.485,0.579),(0.28,0.28),color="k",linewidth=3)


ax.text(0.385,0.28,s="...",fontsize=70)

ax.text(0.415,0.284,s="0 Count",fontsize=55)
ax.text(0.415,0.269,s="1 Count",fontsize=55)

ax.text(0.492,0.284,s="2",fontsize=55)
ax.text(0.492,0.269,s="5",fontsize=55)
ax.plot((0.508,0.508),(0.265,0.295),color="k",linewidth=3)

ax.text(0.515,0.284,s="4",fontsize=55)
ax.text(0.515,0.269,s="3",fontsize=55)
ax.plot((0.531,0.531),(0.265,0.295),color="k",linewidth=3)

ax.text(0.538,0.284,s="0",fontsize=55)
ax.text(0.538,0.269,s="8",fontsize=55)
ax.plot((0.555,0.555),(0.265,0.295),color="k",linewidth=3)

ax.text(0.557,0.284,s="12",fontsize=55)
ax.text(0.562,0.269,s="1",fontsize=55)

ax.text(0.588,0.28,s="...",fontsize=70)


ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(oval1)


ax.set_axis_off()

plt.show()


#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,18))

ax.set_xlim((0.33,0.62))
ax.set_ylim((0.25,0.44))

#ax.set_xlim((0,1))
#ax.set_ylim((0,1))


rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
rect2 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")


rect4 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
rect5 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

oval1 = matplotlib.patches.Ellipse((0.395,0.37),0.06,0.1,color="r",fill=False,linewidth=10)

ax.text(0.37,0.365,s="+",fontsize=70)
ax.text(0.40,0.365,s="+",fontsize=70)

ax.arrow(0.395,0.318,0.0,-0.008,width=0.003,color="k")

rect6 = matplotlib.patches.Rectangle((0.35,0.265),0.094,0.03,color="k",fill=False,linewidth=4)

ax.plot((0.35,0.444),(0.28,0.28),color="k",linewidth=3)


ax.text(0.325,0.28,s="...",fontsize=70)

ax.text(0.455,0.284,s="0 Count",fontsize=55)
ax.text(0.455,0.269,s="1 Count",fontsize=55)

ax.text(0.357,0.284,s="8",fontsize=55)
ax.text(0.357,0.269,s="0",fontsize=55)
ax.plot((0.373,0.373),(0.265,0.295),color="k",linewidth=3)

ax.text(0.38,0.284,s="0",fontsize=55)
ax.text(0.375,0.269,s="11",fontsize=55)
ax.plot((0.397,0.397),(0.265,0.295),color="k",linewidth=3)

ax.text(0.405,0.284,s="0",fontsize=55)
ax.text(0.405,0.269,s="7",fontsize=55)
ax.plot((0.422,0.422),(0.265,0.295),color="k",linewidth=3)

ax.text(0.423,0.284,s="12",fontsize=55)
ax.text(0.428,0.269,s="0",fontsize=55)

ax.text(0.523,0.28,s="...",fontsize=70)


ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(oval1)


ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,18))

ax.set_xlim((0.33,0.59))
ax.set_ylim((0.2,0.44))

#ax.set_xlim((0,1))
#ax.set_ylim((0,1))


# rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.07,0.03,color="#396296")
# rect2 = matplotlib.patches.Rectangle((0.43,0.38),0.04,0.03,color="#1f3fcf")
# rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.09,0.03,color="#ab05e8")


# rect4 = matplotlib.patches.Rectangle((0.36,0.33),0.14,0.03,color="#396296")
# rect5 = matplotlib.patches.Rectangle((0.50,0.33),0.06,0.03,color="#337733")

rect1 = matplotlib.patches.Rectangle((0.36,0.38),0.03,0.03,color="#337733")
rect2 = matplotlib.patches.Rectangle((0.39,0.38),0.08,0.03,color="#396296")
rect3 = matplotlib.patches.Rectangle((0.47,0.38),0.06,0.03,color="#ab05e8")
rect4 = matplotlib.patches.Rectangle((0.53,0.38),0.03,0.03,color="#36cf1f")

rect5 = matplotlib.patches.Rectangle((0.36,0.33),0.12,0.03,color="#36cf1f")
rect6 = matplotlib.patches.Rectangle((0.48,0.33),0.03,0.03,color="#7333bd")
rect7 = matplotlib.patches.Rectangle((0.51,0.33),0.05,0.03,color="#396296")


oval1 = matplotlib.patches.Ellipse((0.405,0.37),0.03,0.1,color="r",fill=False,linewidth=10)


ax.text(0.397,0.365,s="+",fontsize=70)

ax.text(0.399,0.3,s="-",fontsize=120)

rect8 = matplotlib.patches.Rectangle((0.40,0.265),0.01,0.03,color="#396296")


ax.text(0.3975,0.245,s="=",fontsize=70)

rect9 = matplotlib.patches.Rectangle((0.40,0.205),0.01,0.03,color="#36cf1f")




ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
ax.add_patch(rect8)
ax.add_patch(rect9)

ax.add_patch(oval1)


ax.set_axis_off()

plt.show()

#%%
fig,ax = plt.subplots()
fig.set_size_inches((18,4))

ax.set_xlim((-0.1,0.7))
ax.set_ylim((0.88,1))

rect1 = matplotlib.patches.Rectangle((0.00,0.95),0.2,0.03,color="#337733")
rect2 = matplotlib.patches.Rectangle((0.00,0.90),0.2,0.03,color="#36cf1f")

rect3 = matplotlib.patches.Rectangle((0.40,0.95),0.07,0.03,color="#337733")
rect4 = matplotlib.patches.Rectangle((0.47,0.95),0.13,0.03,color="#36cf1f")


rect5 = matplotlib.patches.Rectangle((0.40,0.90),0.07,0.03,color="#36cf1f")
rect6 = matplotlib.patches.Rectangle((0.47,0.90),0.13,0.03,color="#337733")



ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)


ax.text(0.25,0.92,s="vs",fontsize=120)

ax.set_axis_off()

plt.show()