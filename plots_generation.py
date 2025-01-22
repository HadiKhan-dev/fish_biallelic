import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import networkx as nx

warnings.filterwarnings("ignore")
np.seterr(divide='ignore',invalid="ignore")
pass

#%%
def change_labeling(haplotype_data,relabeling_dict):
    """
    Takes as input a list haplotype_data (from the output of 
    generate_haplotypes_all) and changes the labelling of 
    the haplotypes as given by relabeling_dict
    """
    
    new_labeling = []
    
    for i in range(len(haplotype_data)):
        
        hap_dict = haplotype_data[i][3]
        new_hap_dict = {}
        
        for k in hap_dict.keys():
            new_hap_dict[relabeling_dict[(i,k)][1]] = hap_dict[k]
        
        new_hap_dict = dict(sorted(new_hap_dict.items()))
        new_labeling.append([haplotype_data[0],haplotype_data[1],haplotype_data[2],new_hap_dict])
    
    return new_labeling

def nodes_list_to_pos(nodes_list,layer_dist = 4,vert_dist=2):
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
    
    for node_layer in nodes_list:
        for node in node_layer:
            if node[0] not in num_in_layer.keys():
                num_in_layer[node[0]] = 0
            num_in_layer[node[0]] += 1
    
    for node_layer in nodes_list:
        for node in node_layer:
            x_val = node[0]*layer_dist+offset
        
            total_vert_dist = (num_in_layer[node[0]]-1)*vert_dist
        
        
            top_starting = total_vert_dist/2
        
            y_val = top_starting-node[1]*vert_dist
        
            pos_dict[node] = np.array([x_val,y_val])
    
    return pos_dict

def planarify_graph(nodes,edges):
    """
    Takes a list of layers of nodes and a list of edges
    between consecutive layers. Relabels the nodes and edges
    so that the final graph produced when plotted tries to 
    minimise the number of crossing edges
    """
    relabeling_dict = {}
    
    sorted_zero = sorted(edges[0],key=lambda x: x[0][1])
    
    cur_index = 0
    seen_old = set([])
    for edge in sorted_zero:
        if edge[0] not in seen_old:
            seen_old.add(edge[0])
            relabeling_dict[edge[0]] = (0,cur_index)
            cur_index += 1
    for node in nodes[0]:
        if node not in relabeling_dict.keys():
            relabeling_dict[node] = (0,cur_index)
            cur_index += 1
    
    
    for i in range(len(edges)):
        basic_edges = []
        for edge in edges[i]:
            basic_edges.append((relabeling_dict[edge[0]],edge[1]))
        sorted_basics = sorted(basic_edges,key=lambda x: x[0][1])
        
        cur_index = 0
        seen_old = set([])
        
        for edge in sorted_basics:
            if edge[1] not in seen_old:
                seen_old.add(edge[1])
                relabeling_dict[edge[1]] = (i+1,cur_index)
                cur_index += 1
        for node in nodes[i+1]:
            if node not in relabeling_dict.keys():
                relabeling_dict[node] = (i+1,cur_index)
                cur_index += 1
    
    final_edges = []
    
    for i in range(len(edges)):
        adding = []
        for edge in edges[i]:
            new_edge = (relabeling_dict[edge[0]],relabeling_dict[edge[1]])
            adding.append(new_edge)
        final_edges.append(adding)
    
    return (nodes,final_edges,relabeling_dict)
    
def generate_graph_from_matches(matches_list,
                                layer_dist = 4,
                                vert_dist = 2,
                                planarify=False,
                                size_usage_based=False,
                                hap_usages=None
                                ):
    """
    Takes as input a list of two elements: a list of nodes
    and a list of edges between nodes.
    
    Creates a layered networkx graph of the nodes with the
    edges between them
    
    If planarify = True then function tries to create
    a graph that is as planar as possible
    
    If size_usage_based = True then the size of each node
    is proportional to how many haps it gets used in.
    
    In such a case the additional parameter hap_usages must
    be provided
    """
        
    if size_usage_based == True:
        if hap_usages == None:
            assert False,"Block level haplotype usages not provided"
    
    nodes = matches_list[0]
    edges = matches_list[1]
    
    if planarify:
        pdr = planarify_graph(nodes,edges)
        nodes = pdr[0]
        edges = pdr[1]
        rev_map = {v:k for k,v in pdr[2].items()}
    
    num_layers = len(nodes)
    max_haps_in_layer = 0
    for layer in nodes:
        max_haps_in_layer = max(max_haps_in_layer,len(layer))
    
    nodes_pos = nodes_list_to_pos(nodes,layer_dist=layer_dist,vert_dist=vert_dist)
    
    if size_usage_based:
        node_sizes = []
        for block in range(len(nodes)):
            print("BK",block)
            block_sizes = []
            block_dict = {}
            print(nodes[block])
            for full_node in nodes[block]:
                print(full_node)
                node = full_node[1]
                if not planarify:
                    try:
                        block_dict[node] = 1+hap_usages[block][1][node]
                    except:
                        block_dict[node] = 1
                else:
                    new_label = pdr[2][(block,node)][1]
                    try:
                        block_dict[new_label] = 1+hap_usages[block][1][node]
                    except:
                        block_dict[new_label] = 1
            
            for i in range(len(block_dict)):
                block_sizes.append(block_dict[i])
            node_sizes.append(block_sizes)
        flattened_sizes = [4*x for xs in node_sizes for x in xs]
        use_size = flattened_sizes
    else:
        use_size = 600
    
    
    
    flattened_edges = [x for xs in edges for x in xs] #Flatten the edges list
    flattened_nodes = [x for xs in nodes for x in xs]
    
    G = nx.Graph()
    G.add_nodes_from(flattened_nodes)
    G.add_edges_from(flattened_edges)
    
    fig,ax =plt.subplots()
    nx.draw(G,pos=nodes_pos,node_size=use_size)
    ax.set_xlim(left=0,right=layer_dist*num_layers)
    ax.set_ylim(bottom=-0.5*vert_dist*max_haps_in_layer,top=1+0.5*vert_dist*max_haps_in_layer)
    
    # for i in range(num_layers):
    #     ax.text(x=layer_dist*(0.5+i),y=0.5+0.5*vert_dist*max_haps_in_layer,s=f"{i}",horizontalalignment="center")
    #     if i != 0:
    #         ax.axvline(x=layer_dist*i,color="k",linestyle="--")
    
    fig.set_facecolor("white")
    fig.set_size_inches((num_layers,8))
    plt.show()
    
def make_heatmap(probs_list,recomb_rate):
    """
    Takes as input a list of matrices of probabilities for the hidden
    states of a HMM and makes a heatmap for them
    """
    
    def flattenutrm(matrix):
        """
        flatten an upper triangular matrix
        """
        num_rows = matrix.shape[0]
        
        fltr = []
        
        for i in range(num_rows):
            fltr.extend(matrix[i,i:])
        
        return fltr
    
    flattened_list = []
    

    for i in range(len(probs_list)):
        flattened_list.append(flattenutrm(probs_list[i]))
    flattened_array = np.array(flattened_list).transpose()
    
    hap_names = []
    
    num_haps = len(probs_list[0])
    
    
    for i in range(num_haps):
        for j in range(i,num_haps):
            hap_names.append(f"({i},{j})")

    fig,ax = plt.subplots()
    fig.set_size_inches(11,6)
    sns.heatmap(flattened_array,yticklabels=hap_names)
    plt.title(f"Recombination rate: {recomb_rate}")
    plt.show()
    
def make_heatmap_path(best_path,num_haps,recomb_rate):
    """
    Makes a heatmap based on max likelihood path
    """
    
    def flattenutrm(matrix):
        """
        flatten an upper triangular matrix
        """
        num_rows = matrix.shape[0]
        
        fltr = []
        
        for i in range(num_rows):
            fltr.extend(matrix[i,i:])
        
        return fltr
    
    flattened_list = []
    
    for i in range(len(best_path)):
        make_matrix = np.zeros((num_haps,num_haps))
        make_matrix[best_path[i][0],best_path[i][1]] = 1
        flattened_list.append(flattenutrm(make_matrix))
    
    flattened_array = np.array(flattened_list).transpose()
    
    hap_names = []

    for i in range(num_haps):
        for j in range(i,num_haps):
            hap_names.append(f"({i},{j})")

    fig,ax = plt.subplots()
    fig.set_size_inches(11,6)
    sns.heatmap(flattened_array,yticklabels=hap_names)
    plt.title(f"Recombination rate: {recomb_rate}")
    plt.show()