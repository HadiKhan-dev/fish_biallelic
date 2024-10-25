import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
subset_color = [
    "gold",
    "violet",
    "violet",
    "violet",
    "violet",
    "limegreen",
    "limegreen",
    "darkorange",
]

def nodes_list_to_pos(nodes_list,layer_dist = 3,vert_dist=2):
    """
    Takes as input a list of nodes which are a 2-tuple with the
    first element of the tuple signifying the layer and the second
    the identifier within the layer. The second identifer must begin 
    with 0 and count upwards from there for each layer
    
    Returns a list of positions for plotting the layers in networkx
    """
    
    offset = layer_dist/2
    
    pos_dict = {}
    
    num_in_layer = {}
    
    for node in nodes_list:
        if node[0] not in num_in_layer.keys():
            num_in_layer[node[0]] = 0
        num_in_layer[node[0]] += 1
    
    for node in nodes_list:
        x_val = node[0]*layer_dist+offset
        
        total_vert_dist = (num_in_layer[node[0]]-1)*vert_dist
        
        
        top_starting = total_vert_dist/2
        
        y_val = top_starting-node[1]*vert_dist
        
        pos_dict[node] = np.array([x_val,y_val])
    
    return pos_dict

nodes = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(3,0),(3,1),(3,2)]

nodes_pos = nodes_list_to_pos(nodes)
G = nx.Graph()
G.add_nodes_from(nodes)
nx.draw(G,pos=nodes_pos)
plt.show()
#%%
a = [0,1,2,3]
b = []
c = [4,5]
d = [6,7,8]
G = nx.Graph()
G.add_nodes_from(a,layer=0)
#G.add_nodes_from(b,layer=1)
G.add_nodes_from(c,layer=2)
G.add_nodes_from(d,layer=3)
pos = nx.multipartite_layout(G, subset_key="layer")
pos[7] = np.array([-2,-2])

G.add_edges_from([(7,8)])
print(pos)
nx.draw(G,pos)
plt.show()
#%%
def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    print(layers)
    G = nx.Graph()
    for i, lay in enumerate(layers):
        G.add_nodes_from(lay, layer=i**2)
    # for layer1, layer2 in nx.utils.pairwise(layers):
    #     G.add_edges_from(itertools.product(layer1, layer2))
    return G


G = multilayered_graph(*subset_sizes)
#color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
pos = nx.multipartite_layout(G, subset_key="layer")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=False)
plt.axis("equal")
plt.show()