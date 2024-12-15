import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
#%%
def plot_queue(x,y,queue_width,queue_sizes_list):
    
    
    rect_big = patches.Rectangle((x,y),width=queue_width,
                                 height=0.1,color="k",
                                 fill=False)
    
    make_rects = []
    cur_start = x
    
    tot_size_so_far = 0
    
    total_queue_size = sum(queue_sizes_list)
    
    queue_rels = np.array(queue_sizes_list)/total_queue_size
    
    ax.text(x-0.05,y+0.03,s="F",fontsize=30)
    ax.text(x+queue_width+0.03,y+0.03,s="R",fontsize=30)
    
    ax.text(x-0.07,y-0.07,s="PIQ",fontsize=30)
    ax.text(x-0.125,y-0.17,s="Rel PIQ",fontsize=30)
    
    for i in range(len(queue_rels)):
        rect_width = queue_rels[i]*queue_width
        make_rects.append(patches.Rectangle((cur_start,y),width=rect_width,
                                     height=0.1,color="k",
                                     fill=False))
        
        if queue_sizes_list[i] == 0:
            num_digits_in_text = 1
        else:
            num_digits_in_text = 1+math.floor(math.log10(queue_sizes_list[i]))
        ax.text(cur_start+rect_width/2-0.01*num_digits_in_text,y+0.03,s=f"{queue_sizes_list[i]}",fontsize=30)
        
        if tot_size_so_far == 0:
            num_digits_in_piq = 1
        else:
            num_digits_in_piq = 1+math.floor(math.log10(tot_size_so_far))
        ax.text(cur_start+rect_width/2-0.01*num_digits_in_piq,y-0.07,s=f"{tot_size_so_far}",fontsize=30)
            
        num_digits_in_rel_piq = 3.5
        ax.text(cur_start+rect_width/2-0.01*num_digits_in_rel_piq,y-0.17,s=f"{tot_size_so_far/total_queue_size:.2f}",fontsize=30)
            
        
        tot_size_so_far += queue_sizes_list[i]
        
        cur_start += rect_width
        
    for rect in make_rects:
        ax.add_patch(rect)
        
    ax.add_patch(rect_big)
    
#%%
fig,ax = plt.subplots()

fig.set_facecolor("#f2e1d8")
ax.set_facecolor("#f2e1d8")
fig.set_size_inches(18,10)

plot_queue(0.25,0.5,0.5,[5,8,10,3,20])

ax.axis('off')


plt.show()

#%%
fig,ax = plt.subplots()

fig.set_facecolor("#f2e1d8")
ax.set_facecolor("#f2e1d8")
ax.set_xbound(0,1)
ax.set_ybound(0,1)
fig.set_size_inches(8,10)

rect_big = patches.Rectangle((0.3,0.4),width=0.4,
                             height=0.1,color="k",
                             fill=False)
ax.text(0.34,0.43,s="Exchange",fontsize=30)

p1 = patches.Rectangle((0.7,0.7),width=0.1,
                       height=0.1,color="k",fill=False)
p2 = patches.Rectangle((0.85,0.4),width=0.1,
                       height=0.1,color="k",fill=False)
p3 = patches.Rectangle((0.7,0.1),width=0.1,
                       height=0.1,color="k",fill=False)
us = patches.Rectangle((0.1,0.7),width=0.2,
                       height=0.1,color="k",fill=False)

ex_arrow = patches.Arrow(0.69,0.2,-0.18,0.18,
                         width=0.1,color="#4a1000")

us_arrow = patches.Arrow(0.69,0.2,-0.18,0.18,
                         width=0.1,color="#4a1000")

ax.text(0.34,0.43,s="Exchange",fontsize=30)
ax.text(0.71,0.73,s="P1",fontsize=30)
ax.text(0.86,0.43,s="P2",fontsize=30)
ax.text(0.71,0.13,s="P3",fontsize=30)
ax.text(0.16,0.73,s="Us",fontsize=30)


ax.add_patch(rect_big)
ax.add_patch(p1)
ax.add_patch(p2)
ax.add_patch(p3)
ax.add_patch(us)

ax.add_patch(ex_arrow)
#ax.add_patch(us_arrow)

ax.axis('off')


plt.show()