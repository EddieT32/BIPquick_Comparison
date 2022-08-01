"""
Created January, 2022

Author: Edward Tiernan
"""


"""
-------------------------------------------------------------------------------------------------------------------------
Header / Imports
-------------------------------------------------------------------------------------------------------------------------
"""
"""Temp: I'm trying to reconstruct BIPquick so that it accepts a graph object.  
And I'm going to try to create that graph object using subroutines from SPRNTtoMETIS_KaHIP"""

# For handling the array operations
import numpy as np

# Sets the print statements to accomodate large arrays
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# For reporting the time elapsed
import time

# For storing the array as a traversable graph
import networkx as nx

# For parallelizing the graph traversals
import multiprocessing as mp

# For generalizing the paths
import os

# For managing string concatination
import re

# For profiling
import cProfile

# For storing ouput
import pickle as pkl


def path_generator(TRB, partitions):
    global sprnt_file, linknode_output
    parent_folder = r"/home/etiernan/Desktop/METIS_KaHIP_Playground/"
    sprnt_file = os.path.join(parent_folder, "BIPquick_TDR/BIPquick_Code/TRB_SPRNT_Files/" + TRB + "_1hr_interval.spt")
    linknode_output = os.path.join(parent_folder, "Output/" + TRB + "/" + TRB + "_BIPquick" + str(partitions) + ".pkl")
    print(sprnt_file)
    return 


"""
-------------------------------------------------------------------------------------------------------------------------
Hardcoded Cases
-------------------------------------------------------------------------------------------------------------------------
"""


def Case2():
    global is_tree, root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    init_globals()

    graph = nx.DiGraph()
    edge_tuples = [('D', 'C', 100.0), ('C', 'B', 10.0), ('C', 'A', 10.0)]
    graph.add_weighted_edges_from(edge_tuples)
    print(list(graph.nodes))
    is_tree = nx.is_tree(graph)

    create_array_from_graph(graph)

    root_idx_list = graph_roots(graph)

    phantom_node_idx_PARAM = len(list(graph.nodes))
    phantom_link_idx_PARAM = len(list(graph.edges))

    return graph


def Case3():
    global is_tree, root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    init_globals()

    graph = nx.DiGraph()
    edge_tuples = [('D', 'C', 10.0), ('C', 'B', 100.0), ('C', 'A', 100.0)]
    graph.add_weighted_edges_from(edge_tuples)
    print(list(graph.nodes))
    is_tree = nx.is_tree(graph)

    create_array_from_graph(graph)

    root_idx_list = graph_roots(graph)

    phantom_node_idx_PARAM = len(list(graph.nodes))
    phantom_link_idx_PARAM = len(list(graph.edges))

    return graph


def OPTIMAL_tree():
    global is_tree, root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    init_globals()

    graph = nx.DiGraph()
    edge_tuples = [("H", "G"), ("G", "F"), ("F", "Q"), ("Q", "P"), ("Q", "S"), ("F", "E"), ("E", "O"), ("E", "D"), ("D", "J"), ("J", "R"), ("D", "C"), ("C", "B"), \
        ("B", "I"), ("B", 'A'), ("E", "N"), ("N", "M"), ("M", "L"), ("M", "K")]
    graph.add_edges_from(edge_tuples, weight=1.0)
    print(list(graph.nodes))
    is_tree = nx.is_tree(graph)

    create_array_from_graph(graph)

    root_idx_list = graph_roots(graph)

    phantom_node_idx_PARAM = len(list(graph.nodes))
    phantom_link_idx_PARAM = len(list(graph.edges))

    return graph


def OPTIMAL_cross():
    global is_tree, root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    init_globals()

    graph = nx.DiGraph()
    edge_tuples = [("H", "G"), ("G", "F"), ("F", "Q"), ("Q", "P"), ("Q", "O"), ("F", "E"), ("E", "O"), ("E", "D"), ("D", "J"), ("J", "R"), ("D", "C"), ("C", "B"), \
        ("B", "I"), ("B", 'A'), ("E", "N"), ("N", "M"), ("M", "L"), ("M", "K")]
    graph.add_edges_from(edge_tuples, weight=1.0)
    print(list(graph.nodes))
    is_tree = nx.is_tree(graph)

    create_array_from_graph(graph)

    root_idx_list = graph_roots(graph)

    phantom_node_idx_PARAM = len(list(graph.nodes))
    phantom_link_idx_PARAM = len(list(graph.edges))

    return graph


node_num_tracker = []

""" 
-------------------------------------------------------------------------------------------------------------------------
BIPquick Main Subroutine
-------------------------------------------------------------------------------------------------------------------------
""" 

# The bipquick routine controls the calls to subsequent routines 
def bipquick(graph, partitions):

    # These variables must be passed to other subroutines not called from bipquick()
    global nodes, links, partition_threshold, accounted_for_edges, case_three, effective_root_name, \
        root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    # path_generator(TRB)

    create_array_from_graph(graph, partitions)
    print(len(list(graph.nodes)))
    print(linknode_output)

    root_idx_list = graph_roots(graph)

    phantom_node_idx_PARAM = len(list(graph.nodes))
    phantom_link_idx_PARAM = len(list(graph.edges))


    # This for-loop drives the Branchwise ITERATIVE Partitioning
    start = time.time()

    for ii in range(partitions):
        if ii % 5 == 0:
            print("Partition    ", ii)

        # HACK - write the last-processor bypass u dummy
        if ii == partitions - 1:
            print("Last partition bypass reached")
            nodes_unassigned = np.where(nodes[:, ni_Image] == None)[0]
            nodes[nodes_unassigned, ni_Image] = ii
            links_unassigned = np.where(links[:, li_Image] == None)[0]
            links[links_unassigned, li_Image] = ii
            break

        # The cumulative weight for each node needs to be reset after each subnetwork is taken out
        nodes[:, nr_totalweight_u] = 0.0

        # Initialize the global boolean needed for traverse_partition_subnetwork
        case_three = False

        # Determine the new cumulative weight for each node for a given network
        calc_totalweight_assigner()
        partition_threshold = max_weight / (partitions - ii)
        # print("The max weight/partition threshold is:   ", max_weight, "/", partition_threshold)

        # The effective_root is the root where the partition_threshold is found exactly.  Alternatively, if the
        # partition_threshold isn't found at a node or spanned by a link, then the effective_root is the nearest
        # overestimate of the partition_threshold by the cumulative node weight.
        effective_root = seek_ideal_check()
        effective_root_name = nodes[effective_root, node_name]

        # print("The effective root is:", effective_root_name, nodes[effective_root, :])

        # If the partition_threshold exists at a node, traverse the graph upstream of that node and assign image numbers
        if ideal_exists:

            # This represents Case 1: The partition threshold is found at an existing node
            print("Case 1 Found! " + effective_root_name, effective_root)
            traverse_partition_subnetwork(effective_root_name, case_three, ii)
            partitioned_links(ii)

        # Otherwise, check to see if the partition_threshold is spanned by any link
        else:
            spanning_link = seek_threshold_spanning(partition_threshold)

            # This represents Case 2: The partition threshold is spanned by a link
            # if spanning_link != '':
            #     print("Case 2 Found ", links[spanning_link, :])

            effective_root_check = ''
            # If the partition threshold is not spanned, the return will be an empty string
            effective_root_iter = 0

            while spanning_link == '':
                # print("Hit the While ------------------------------------------------------")
                case_three = True

                if effective_root != effective_root_check:
                    effective_root_check = effective_root
                    effective_root_iter = 0
                else:
                    effective_root_iter = effective_root_iter + 1
                    # print("Effective Root & Iter", nodes[effective_root], effective_root_iter)

                # This represents Case 3: The partition threshold is not found at an existing node or spanned by a link
                # print("Case 3 Found", nodes[effective_root])

                """
                For Case 3, the effective_root will be the node whose totalweight is the nearest overestimate
                of the partition threshold. Then, an upstream node from the effective node (chosen arbitrarily)
                is used as the root for a travers_partition_subnetwork() call.  The effective root totalweight
                is reduced by the upstream node totalweight + the link length
                """

                # The link upstream of the effective root is chosen
                try:
                    lrows = np.where(links[:, li_Mnode_d] == effective_root_name)[0]
                except:
                    print(effective_root_name, "isn't found in the links li_Mnode_d column")
                # print("Upstream Link of Effective Root", links[lrows])

                # The upstream link length, upstream node, and upstream node totalweight are determined
                upstream_link_length = links[lrows[effective_root_iter], lr_Length]
                upstream_node_name = links[lrows[effective_root_iter], li_Mnode_u]
                try:
                    upstream_node_idx = np.where(nodes[:, node_name] == upstream_node_name)[0][0]
                except:
                    print(upstream_node_name, "isn't in the nodes array")
                    quit()
                upstream_weight = nodes[upstream_node_idx, nr_totalweight_u]
                # print("Upstream link length:", upstream_link_length)
                # print("Upstream node:", nodes[upstream_node_idx])

                # The upstream node weight and upstream link length are used to reduce the effective root totalweight
                total_clipped_weight = float(upstream_weight) + weighting_function(float(upstream_link_length))
                # print("Total clipped weight", total_clipped_weight)

                # The effective root totalweight is reduced by the total clipped weight
                # effective_root = int(effective_root)
                nodes[effective_root, nr_totalweight_u] = nodes[effective_root, nr_totalweight_u] - total_clipped_weight
                # print("Nodes[effective root]", nodes[effective_root, :], total_clipped_weight)

                # The partition threshold is also reduced by the total clipped weight
                partition_threshold = partition_threshold - total_clipped_weight
                # print("Partition threshold", partition_threshold)

                # The subgraph deriving from the upstream node is traversed and assigned
                traverse_partition_subnetwork(upstream_node_name, case_three, ii)

                # print("Graph")
                # print(graph.edges.data('weight'))
                # print(graph.nodes)
                # quit()
            
                # After the graph has been reduced, check again for a spanning link and effective root
                spanning_link = seek_threshold_spanning(partition_threshold)
                effective_root = seek_ideal_check()
                effective_root_name = nodes[effective_root, node_name]

                # if effective_root_iter == 1:
                #     print(graph.nodes)
                #     print(spanning_link, partition_threshold)
                #     print("The effective root is:", nodes[effective_root])

            """
            Once the spanning link is not empty, Case 3 transforms into a Case 2.
            A phantom node must be induced in the spanning link, transforming Case 2 into Case 1
            """
            if case_three:
                # print("The partition has made it out of Case 3")
                case_three = False


            # The distance from the upstream node is determined
            length_from_start = linear_interpolator(partition_threshold, spanning_link)

            # A phantom node is created in the spanning_link
            phantom_node_generator(spanning_link, length_from_start, ii)

            # # The graph must be recreated to account for the change in topology from adding a phantom node
            # graph = create_graph()

            # The subgraph deriving from the phantom node is traversed and assigned
            traverse_partition_subnetwork(nodes[phantom_node_idx, node_name], case_three, ii)

            partitioned_links(ii)
                
        print("----------------------------------------------")

    # This marks the end of one iterative partition assignment
    nx_profiler_end = time.time()
    total_time = nx_profiler_end - start
    nodes[:, nr_totalweight_u] = 0.0
    check_boundary_nodes()

    with open(linknode_output, 'wb') as time_node_link_file:
        pkl.dump([total_time, nodes, links], time_node_link_file)
    # print(nodes)
    # print(links)
    connectivity = connectivity_metric()
    part_size_balance = part_size_balance_metric(partitions)[0]


    """End of Partition Summary"""
    print("===================================================================")
    print("System:", TRB, "Partitions:", partitions)
    print("Partition Time:", nx_profiler_end - start)
    print("Connectivity:", connectivity)
    print("Balance:", part_size_balance)
    print("===================================================================")
    return


"""
Initialization:

This section defines the functions that initialize and populate the two governing arrays, nodes and links.

init_globals():             this routine initializes the globals that are used by several other subroutines

read_sprnt_for_counters():  this routine reads the sprnt file for counters

read_sprnt_contents():      this routine reads the sprnt file again and populates the created arrays

junction_collapsing():      this routine processes the disjoint junction standard from SPRNT so the graph is contiguous

create_graph():              this routine creates a networkX graph object from the link/node arrays

graph_is_tree():            this function determines if the network is a tree structure
"""

def init_globals():

    # Node array globals
    global nodes, ni_idx, node_name, nr_totalweight_u, ni_Image, ni_is_Boundary
        
    # Link array globals
    global links, li_idx, li_Mnode_u, li_Mnode_d, lr_Length, li_Image, lr_Target
    
    # Partition output array globals
    global net_ni_idx, net_ni_image, net_ni_is_edge, net_li_idx, net_li_image

    # Calculated global variables
    global max_weight, partition_threshold, ideal_exists

    # SPRNT globals
    global node_id, segment_up, segment_down, segment_length, junction_down, junction_up1, junction_coeff1, \
        junction_up2, junction_coeff2

    ni_idx = 0  # the node index (i.e. its position in the array)
    node_name = 1
    nr_totalweight_u = 2  # the cumulative weight of all links upstream
    ni_Image = 3
    ni_is_Boundary = 4

    li_idx = 0 # the link index 
    li_Mnode_u = 1  # ID of upstream node
    li_Mnode_d = 2  # ID of downstream node
    lr_Length = 3  # length of the link (0 if type != CONDUIT)
    li_Image = 4
    lr_Target = 1.0  # For the time being, the target length of an element is a hardcoded parameter

    net_ni_idx = 0 # the node index in the Partition Output array
    net_ni_image = 1 # the image (i.e. partition) number to which that node has been assigned
    net_ni_is_edge = 2 # the number of partitions that this node borders (besides its own)
    net_li_idx = 0 # the link index in the Partition Output array
    net_li_image = 1 # the image (i.e. partition) number to which that node has been assigned

    max_weight = 0.0 # the largest totalweight in the current graph (guaranteed to be at one of the outfalls)
    partition_threshold = 0.0 # the totalweight value associated with a single partition
    ideal_exists = False # a boolean describing whether or not a network is Case 1

    node_id = 0

    segment_up = 0
    segment_down = 1
    segment_length = 2

    junction_down = 0
    junction_up1 = 1
    junction_coeff1 = 2
    junction_up2 = 3
    junction_coeff2 = 4
    return


def read_sprnt_for_counters():
    """
    This function inspects the contents of the SPRNT input file and creates correctly sized arrays for each of the data
    types.

    This allows us to use np.arrays rather than 2D lists for nodes_name, nodes_geo, segments, and junctions.
    lateralsources and qsources need to be 2D lists because of the inconsistent timeseries entries

    The second dimension of the arrays is hard-coded to reflect that each data type in SPRNT contains a different number
     of parameters.

    In the San Antonio - Guadalupe River basin the SPRNT input file is 3.8 millions lines.
    """
    global total_count, null_count, node_count, segment_count, lateral_count, junction_count, qsource_count, \
        nodes_name, nodes_geo, segments, lateralsources, junctions, qsources, boundaryconditions

    total_count = len(sprnt_contents)
    null_count, node_count, segment_count, lateral_count, junction_count, qsource_count = 0, 0, 0, 0, 0, 0

    for ii in range(total_count):
        if sprnt_contents[ii].find("#") != -1:
            null_count = null_count + 1
        elif sprnt_contents[ii].find('node') != -1:
            node_count = node_count + 1
        elif sprnt_contents[ii].find("segment") != -1:
            segment_count = segment_count + 1
        elif sprnt_contents[ii].find("lateralsource") != -1:
            lateral_count = lateral_count + 1
        elif sprnt_contents[ii].find("junction") != -1:
            junction_count = junction_count + 1
        elif sprnt_contents[ii].find("qsource") != -1:
            qsource_count = qsource_count + 1

    # The sizes of these arrays is network dependent, may need to write some functions to generalize them
    # sys.setrecursionlimit(node_count)
    nodes_geo = np.empty((node_count, 2), dtype=object)
    nodes_name = np.empty((node_count, 6), dtype=object)
    segments = np.empty((segment_count, 3), dtype=object)
    #lateralsources = np.empty((lateral_count, max_time), dtype=object)
    lateralsources = []
    lateralsources.append([])
    junctions = np.empty((junction_count, 5), dtype=object)
    #qsources = np.empty((qsource_count, max_time), dtype=object)
    qsources = []
    qsources.append([])
    # The boundary condition is hard-coded with the value obtained from the SPRNT input file
    # boundaryconditions = (root_raw, root_bc)

    return


def read_sprnt_contents():
    """
    This function populates the various data type arrays with the values parsed from the SPRNT input file database.

    The function loops through each line in the database; when it finds the keyword that indicates a certain
    data-type, the corresponding values are added to the np.array.  This function is sort of messily split between
    keyword-value searches and assumptions of .spt line organization.

    lateralsources and qsources are added differently.  Each time a new time-value pair is found it is appended to
    the 2nd dimension of the 2D list.
    """

    node_number, segment_number, lateralsource_number, junction_number, qsource_number = 0, 0, 0, 0, 0
    for line in range(len(sprnt_contents) - 1):

        # If the line is commented out, ignore it
        if sprnt_contents[line].find("#") != -1:
            continue

        # When you find a node definition - do this
        # Need to replace the nodes population with the AY-transform
        elif sprnt_contents[line].find("node") != -1:
            #print("Found Node")
            splitline = sprnt_contents[line].split()
            for elem in range(len(splitline)):
                try:
                    keyword = splitline[elem].split('=')
                    if keyword[0] == 'id':
                        nodes_name[node_number, node_id] = keyword[1]
                    if keyword[0] == 'sR':
                        nodes_name[node_number, node_sR] = keyword[1]
                    if keyword[0] == 'n':
                        nodes_name[node_number, node_n] = keyword[1]
                    if keyword[0] == 'zR':
                        nodes_name[node_number, node_zR] = keyword[1]
                    if keyword[0] == 'hR':
                        nodes_name[node_number, node_hR] = keyword[1]
                except:
                    continue
            node_num_tracker.append(node_number)
            node_number = node_number + 1

        # When a segment definition is found - do this
        elif sprnt_contents[line].find("segment") != -1:
            #print("Found Segment")
            spline = re.split(' up=| down=| length=| ', sprnt_contents[line])
            segments[segment_number] = spline[2:5]
            segment_number = segment_number + 1

        # When a lateralsource is defined - add the time series
        elif sprnt_contents[line].find("lateralsource") != -1:
            #print("Found Lateral Source")
            if lateralsource_number > 0:
                lateralsources.append([])
            spline = re.split("=|\n", sprnt_contents[line + 1])
            lateralsources[lateralsource_number].append(spline[1])
            nextspline = re.split("=|\n", sprnt_contents[line + 3])
            lateralsources[lateralsource_number].append(nextspline[1])
            temp_line = line + 4
            # Each time series has the source, timeunit, time_number series.  So the time_number series starts at 2
            while sprnt_contents[temp_line].find('t=') != -1:
                timespline = re.split("=|\n| ", sprnt_contents[temp_line])
                for ii in range(len(timespline)):
                    if timespline[ii] == 't':
                        lateralsources[lateralsource_number].append(timespline[ii+1])
                    if timespline[ii] == 'v':
                        lateralsources[lateralsource_number].append(timespline[ii+1])
                temp_line = temp_line + 1
            lateralsource_number = lateralsource_number + 1


        # When a junction is found - add the connecting nodes and coefficients
        elif sprnt_contents[line].find("junction") != -1:
            #print("Found Junction")
            spline = re.split("down=| up1=| coeff1=| up2=| coeff2=|\n", sprnt_contents[line + 1])
            spline = spline[1:-1]
            for ii in range(len(spline)):
                if spline[ii].find(',') != -1:
                    spline[ii] = spline[ii].strip(',')
            try:
                junctions[junction_number] = spline
            except ValueError:
                junctions[junction_number, junction_down:junction_coeff1 + 1] = spline
            junction_number = junction_number + 1

        # When a qsource is found - add the time series
        elif sprnt_contents[line].find("qsource") != -1:
            #print("Found Qsource")
            if qsource_number > 0:
                qsources.append([])
            spline = re.split("=|\n", sprnt_contents[line + 1])
            qsources[-1].append(spline[1])
            nextspline = re.split("=|\n", sprnt_contents[line + 3])
            qsources[-1].append(nextspline[1])
            temp_line = line + 4
            # Similarly to lateral sources, the qsource row has source, timeunit, time_number series data
            while sprnt_contents[temp_line].find('t=') != -1:
                timespline = re.split("=|\n| ", sprnt_contents[temp_line])
                #qsources[qsource_number, time_number] = float(timespline[-2])
                for ii in range(len(timespline)):
                    if timespline[ii] == 't':
                        qsources[-1].append(timespline[ii+1])
                    if timespline[ii] == 'v':
                        qsources[-1].append(timespline[ii+1])
                temp_line = temp_line + 1

            qsource_number = qsource_number + 1

        sum_objects = (node_number + segment_number + lateralsource_number + junction_number + qsource_number)
        if (sum_objects) % 10000 == 0:
            print(sum_objects, "found in read_sprnt_contents")
    return


def junction_collapsing():
    """
    This function relates to the SPRNT data structure of multiple nodes existing in the same physical location.  In
    SPRNT these nodes are connected relationally using a junction object.  However, in SWMM a junction and a node are
    synonymous, and only one node/junction can exist in one point in space.

    The junction's downstream node (i.e. the upstream node for the segment LEAVING the junction) is the node that is
    chosen to persist.  The junction's upstream nodes (i.e. the downstream nodes for the segments ENTERING the junction)
    are removed and the downstream nodes for the entering segments are reassigned.
    """

    for segment in range(len(segments)):
        from_node_id = segments[segment, segment_up].split('_')[-1]
        to_node_id = segments[segment, segment_down].split('_')[-1]
        if len(from_node_id) > 6:
            segments[segment, segment_up] = from_node_id
        if len(to_node_id) > 6:
            segments[segment, segment_down] = to_node_id
        if to_node_id == 'J':
            for junctionnode in range(len(junctions)):
                if (junctions[junctionnode, junction_up1] == segments[segment, segment_down]) or \
                        (junctions[junctionnode, junction_up2] == segments[segment, segment_down]):
                    segments[segment, segment_down] = junctions[junctionnode, junction_down]
    return


def graph_is_tree(graph):
    global is_tree
    is_tree = nx.is_tree(graph)
    return


def create_graph():
    global graph
    graph = nx.DiGraph()

    for segment in segments:
        up_seg = segment[segment_up]
        dn_seg = segment[segment_down]
        length = float(segment[segment_length])
        # if (dn_seg, up_seg) in accounted_for_edges.edges:
        #     continue
        graph.add_edge(dn_seg,up_seg,weight=length)
    return graph


def graph_roots(graph):
    global downstream_roots_idx
    downstream_roots_idx = []
    for node in graph.nodes:
        # print(list(graph.predecessors(node)))
        if list(graph.predecessors(node)) == []:
            root_idx = np.where(nodes[:, node_name] == node)
            downstream_roots_idx.append(root_idx[0][0])

    return downstream_roots_idx


def create_array_from_graph(graph, partitions):
    global nodes, links

    # These arrays are the output from BIPquick, they contain the node-link indices, the image numbers, and the connectivity count
    nodes = np.empty((len(list(graph.nodes)) + partitions - 1, 5), dtype=object)
    links = np.empty((len(list(graph.edges)) + partitions - 1, 5), dtype=object)

    nn = 0
    for node in graph.nodes:
        nodes[nn, ni_idx] = nn
        nodes[nn, node_name] = node
        nodes[nn, ni_is_Boundary] = 0
        nn = nn + 1
    
    ll = 0
    for link in graph.edges:
        links[ll, li_idx] = ll
        links[ll, li_Mnode_d] = link[0]
        links[ll, li_Mnode_u] = link[1]
        links[ll, lr_Length] = graph.edges[link]['weight']
        ll = ll + 1


    return


def initialization():
    """These three functions serve to read and pre-process the SPRNT system so that it can be converted into a NX graph"""
    global graph, root_idx_list, phantom_node_idx_PARAM, phantom_link_idx_PARAM

    init_globals()

    read_sprnt_for_counters()

    read_sprnt_contents()

    junction_collapsing()
    # print("The junctions have been collapsed")

    graph = create_graph()

    graph_is_tree(graph)
    # print(is_tree)


    # print(phantom_link_idx_PARAM + phantom_node_idx_PARAM)

    return graph


def weighting_function(link_length):  
    """ The weight attributed to each link (that will ultimately be assigned to the
        downstream node) are normalized by lr_Target.  This gives an estimate of
        computational complexity. In the future lr_Target can be customized for each link."""
    # A constant for now, but a link attribute in the future
    global lr_Target 

    # The weight ascribed to the node is the number of expected elements on that link
    return link_length/lr_Target 


def netX_upstream_weight_calculation(root):
    """ This method is called in parallel using the Multiprocessor.Pool.map() functionality. It accepts
    a root node argument, then uses networkX functions to determine the subgraph that exists upstream of
    that root node.  The accumulated weight to be attributed to the root node is the sum of the edge weights
    in the subgraph.  Different networkX functions are used for different network types (for speed).
    """

    # The root argument from calc_totalweight_assigner() is an integer, but the graph is in strings.
    root = str(root)

    # Set the accumulated_weight for that root to 0.0
    accumulated_weight = 0.0

    # If the network is a tree structure (from graph_check())
    if is_tree == True:

        # Iterate through the edge tuples that are returned by the nx.bfs_predecessors() function
        for ee in nx.bfs_predecessors(graph, root):

            # Grab the link_weight attribute from the graph edge
            link_weight = float(graph.edges[(ee[1], ee[0])]['weight'])

            # The accumulated weight is the running sum of all link_weights for the upstream subgraph
            accumulated_weight = accumulated_weight + weighting_function(link_weight)

    # If the network is NOT a tree structure (i.e. has cross connections)
    else:

        # A subgraph must be initialized containing just the nodes returned by nx.dfs_postorder_nodes()
        subgraph_root = graph.subgraph(nx.dfs_postorder_nodes(graph, root))

        # Iterate through the edges from this subgraph (which mirror the edges from the OG graph)
        for edge in subgraph_root.edges:

            # Grab the link_weight attribute from the graph edge
            link_weight = float(graph.edges[edge]['weight'])

            # The accumulated weight is the running sum of all link_weights for the upstream subgraph
            accumulated_weight = accumulated_weight + weighting_function(link_weight)
    
    # This subroutine returns a tuple such that the accumulated weight is associated with the root
    return root, accumulated_weight


def calc_totalweight_assigner():
    """This routine is parallelized to allow the totalweight to be calculated for multiple nodes at a time.
    The global networkX graph is traversed by netX_upstream_weight_calculation() for each node, and the 
    accumulated weight is mapped to the nodes that have been visited (this routine only visits nodes that are 
    still in the graph, i.e. excluding the ones that have already been removed)
    """
    global max_weight
    # print("Entering calc_totalweight_assigner()")
    
    # Initializes/Resets the max_weight (used to calculate the partition threshold)
    max_weight = 0

    # Initializes the multiprocessor.Pool object with the number of threads equal to the computer's cpu count
    pool = mp.Pool(mp.cpu_count())
    # pool = mp.Pool(1)


    # The accumulated_weight_list is a list of tuples that contains the output from the parallel pool.map()
    # The list contains the (root, accumulated weight) pairs that are returned from the netX_upstream_weight_calculation()
    # The root nodes that are passed to the netX_upstream_weight_calculation are the nodes that are still in the graph.nodes
        # For some reason it doesn't work if I just call them straight from the graph.nodes
    accumulated_weight_list = pool.map(netX_upstream_weight_calculation, [root for root in nodes[:, node_name] if root in graph.nodes])
    pool.close()

    # Iterates through the accumulated_weight_list tuples
    for pair in accumulated_weight_list:

        # The first element of the tuple is the node
        # node_name = int(pair[0])

        node_index = np.where(nodes[:, node_name] == pair[0])

        # The second element is the totalweight calculated for that node
        nodes[node_index[0][0], nr_totalweight_u] = pair[1]


    # The max_weight is taken as the maximum accumulated weight for the existing nodes
    max_weight = 0
    for root in root_idx_list:
        max_weight = max_weight + nodes[root,nr_totalweight_u]

    # print("Exiting nr_totalweight_assigner, the Max Weight is: ", max_weight)
    return



"""
Subnetwork Identification and Optimal Partitioning:

Contains the functions that separate the nodes and lists arrays into multiple sub-arrays

Global Variables: subnetwork_container_nodes, subnetwork_container_links
    These variables are 3D zeros arrays that are initialized to be (depth x height x width) where 'depth' is the number
    of processors (i.e. the number of subsystems) to which the program will ascribe data, 'height' is the number of
    nodes or links, and 'width' contains the node/link info the same as the nodes, links arrays from the Array
    Initialization section.

subnetwork_carving(root):   

subnetwork_links(): the subnetwork delineation is conducted on a node-wise basis of weights, but the vast majority of
                    the future elements will be contained within the links.  This function adds all of the links that
                    are "contained" within the subsystem to subnetworks_container_links.  Here, "contained" is defined
                    as having both endpoint nodes within the subsystem.

seek_ideal_check():    Computes the partition threshold and checks the network to determine if an optimal partition
                            is possible.  An optimal partition occurs if the partition threshold is found at its exact
                            value at a node.

All above functions can be used in the event that an optimal partition is possible for the network. Should an optimal
partition exist, the subnetwork_carving() routine is called to delineate the subnetwork upstream of that node.
Subsequently, the new network is re-weighted using local_node_weighting() and nr_totalweight_assigner().  Then the
seek_ideal_check is called to determine if optimal partitioning is still possible.  Each is called in main().
"""


def traverse_partition_subnetwork(root, casethree, ii):
    """Given a root node, this function identifies the subgraph that is upstream of this node.
    Then it tags that node with the current image (within the larger partitions loop) and whether or not
    the node is at the boundary of another partition (if the node has already been tagged, it is at a boundary).
    Finally, the nodes and edges in the subgraph are removed from the larger graph.
    """
    # print("In Subnetwork_Carving")

    # To make the parallelization work, the graph and accounted_for_edges must be globals
    global graph, accounted_for_edges

    # The subgraph induced at the root of the OG graph
    subgraph_root = graph.subgraph(nx.dfs_postorder_nodes(graph, root))
    # print(subgraph_root.nodes)

    # The nodes of the subgraph are iterated through
    for node in subgraph_root.nodes:
        try:
            nn = np.where(nodes[:, node_name] == node)[0][0]
        except:
            print(node, "is in the graph but not the nodes array")
            quit()

        # If the node has not yet been assigned (i.e. net_ni_idx == None)
        if nodes[nn, ni_Image] == None:

            # Then assign that row in network_nodes_image as node, image number, is_edge = 0
            nodes[nn, (net_ni_idx, ni_Image)] = nn, ii
        
        # If the node HAS been assigned
        # else:

        #     #Don't change the nodes image, but iterate its ni_is_edge by 1
        #     nodes[nn, ni_is_Boundary] = nodes[nn, ni_is_Boundary] + 1
    
    # Initialize/reset the temporary graph that houses the graph that will be removed
    graph_for_removal = nx.DiGraph()

    # The edges from the subgraph are added to the removal graph and the larger running tally graph
    graph_for_removal.add_edges_from(subgraph_root.edges)
    accounted_for_edges.add_edges_from(subgraph_root.edges)

    if case_three:
        graph_for_removal.add_edges_from([(effective_root_name, root)])
        accounted_for_edges.add_edges_from([(effective_root_name, root)])

    # Subtract the removal graph from the network graph
    graph.remove_edges_from(graph_for_removal.edges)
    # print(network_nodes_image)

    # Check to ensure that there aren't any stranded nodes left in the graph with no neighbors
    graph.remove_nodes_from(list(nx.isolates(graph)))
    # print(list(nx.isolates(graph)))
    # print(graph.edges.data('weight'))
    # quit()
    return


def seek_ideal_check():
    #print("Entering seek_ideal_check()")
    """
    This function determines whether an optimal partition is possible.

    The nodes array is searched for an index containing exactly this threshold value, returning the node_id if it is
    found.  Otherwise, the routine searches for the node with the nearest overestimate of the weight threshold.  The
    nearest overestimate weight is initialized as the max_weight*1.1, as this is guaranteed to both exist and be larger than
    the partition threshold.  In the same loop as the search for the ideal threshold, if the node's cumulative upstream
    weight value is greater than the threshold but less than the current nearest overestimate, the nearest overestimate
    is overwritten.  Should no ideal partition node be found the function exits, passing the node_id of the nearest
    overestimate.

    The end result of this function is that the ideal_exists check is either True or False.  If it is True, that means
    an ideal partition exists and the function passes the id of that root node. If it is False, then an ideal check does
    not exist and the nearest_overestimate_node is passed.  The need for this nearest overestimate will be explained in
    the Non-Ideal Partitioning section.
    """

    # The max_weight and partition_threshold are calculated outside of this routine
    global max_weight, partition_threshold, ideal_exists

    # Initialize/Reset the ideal_root and nearest_overestimate_node to empty strings
    ideal_root, nearest_overestimate_node = '', ''

    # Initialize/Reset the ideal_exists boolean to false
    ideal_exists = False

    # The nearest_overestimate is initialized as the max_weight + some tolerance for numerical precision
    nearest_overestimate = max_weight*1.1
 
    # Iterates through the nodes looking at the totalweight
    for node in graph.nodes:
        nodes_row = np.where(nodes[:, node_name] == node)[0][0]

        # If the totalweight equals the partition threshold
        if nodes[nodes_row, nr_totalweight_u] == partition_threshold:

            # The ideal_root equals the node whose totalweight equals the partition threshold
            ideal_root = nodes[nodes_row, ni_idx]

            # The existence of an ideal_root is set to true
            ideal_exists = True

            # And the ideal_root is returned
            return ideal_root

        # Otherwise, the nodes whose weights are greater than the partition threshold are searched.  If the weight is
        # greater than the partition_threshold AND less than the current nearest_overestimate, the nearest_overestimate
        # is overwritten.
        if (float(nodes[nodes_row, nr_totalweight_u]) > partition_threshold) and \
                (nodes[nodes_row, nr_totalweight_u] < nearest_overestimate):
            
            # The nearest overestimates iteratively reduces to some node whose totalweight is the floor of the range
            # [partition_threshold, max_weight]
            nearest_overestimate = nodes[nodes_row, nr_totalweight_u]
            nearest_overestimate_node = nodes[nodes_row, ni_idx]
            # print("The effective root is:   ", nearest_overestimate_node, "     The totalweight is:     ", nearest_overestimate)
    
    # The nearest_overestimate_node is returned (this is used in Case 3 as the effective_root)
    return nearest_overestimate_node


"""
Non-Ideal Partitioning:

In the likely case that optimal partitioning is not possible (i.e. the partition threshold cannot be found exactly
at any node in the network), another partitioning strategy must be pursued.  As a check, seek_threshold_spanning() is
called to determine if the partition threshold is spanned by any link in the dummy network.

The node weights (which should maintain their values from the larger network) are searched this time for the
nearest underestimate of the partition threshold. Calling subnetwork_carving() on this node yields the first pass at
filling the current subnetwork to the appropriate size. The partition threshold is reduced by the weight on the root
node, and the dummy network is re-assigned weight to reflect the clipped out subnetwork component.

seek_threshold_spanning():      if an ideal partition does not exist it might be possible that a link spans the partition threshold.
                                'Spanning' here means that the weight at the upstream node is less than the threshold, but the sum
                                of the upstream node and a downstream link are greater than the threshold.  This indicates that a
                                phantom node can be placed along that link such that the weight at the phantom node is exactly the
                                value of the partition threshold.

linear_interpolator():          this simple function will be used to determine location of the phantom node along a link

phantom_naming_convenction():   when BIPQuick is translated into Fortran, its not possible to have different data types
                                cohabit the same array. For this reason, the phantom nodes and links will simply be
                                named in numerical order, starting from a value at least one order of magnitude greater
                                than the largest index.

phantom_node_generator():       this function amends the global nodes and links arrays adding 1 phantom node and 1 phantom
                                link to subdivide the link that spans the partition threshold.  This function uses
                                linear_interpolator() to determine the length of the phantom link, populates the *_id, *_idx
                                and upstream/downstream neighbor columns, while assigning default values to the other cells.
"""


def seek_threshold_spanning(partition_threshold):
    # print("Entering seek_threshold_spanning()")
    """
    This function is given a weighted network and a partition threshold value that does not exist at any node.  The
    function's job is to determine whether any link 'spans' the partition threshold, by establishing a range of weights.
    The minimum of this range is the weight value that exists at the upstream node, and the maximum of the range is
    given by the upstream node weight plus the weight contributed by the link itself.

    Once properly populated, the weight_array is compared against the partition threshold argument to determine if any
    links span the threshold.  If so, then that link index is returned.  If not, the function returns an empty string.
    """

    # The variables here are used by other routines in the non-ideal partitioning arm of the algorithm
    global weight_range, spanning_node_upstream_idx, accounted_for_edges

    # Initializes/Resets the spanning_link to an empty string
    spanning_link = ''

    # Initializes the weight_range array as a numpy array of zeroes
    nones = list(links[:,li_idx]).count(None)
    weight_range = np.zeros((len(links) - nones, 2))

    # Iterates through the links array
    for link_row in range(len(weight_range)):

        # The upstream node for that link is found
        upstream_node_name = links[link_row, li_Mnode_u]
        downstream_node_name = links[link_row, li_Mnode_d]
        edge = (downstream_node_name, upstream_node_name)
        # print("Edge", edge)
        # print(graph.edges)
        # quit()
        try:
            upstream_node_idx = np.where(nodes[:,node_name] == upstream_node_name)[0][0]
        except:
            print(upstream_node_name, " does not exist in the nodes array")
            quit()

        # Check if this link_row represents an edge that has already been accounted for
        if accounted_for_edges.has_edge(*edge):
            # print("Found an accounted for edge", upstream_node_name, downstream_node_name)
            continue

        # Its totalweight marks the minimum of that link's weight_range
        weight_range[link_row, 0] = nodes[upstream_node_idx, nr_totalweight_u]
    
        # The weighting function of the current link's length determines the max of the link's weight range
        link_weight = weighting_function(float(links[link_row, lr_Length]))
        weight_range[link_row, 1] = weight_range[link_row, 0] + link_weight
        
    # Once populated, iterate throught the weight_range array
    for weight_row in range(len(weight_range)):

        # Check if the partition_threshold is between the min-max weight for any link
        if (weight_range[weight_row, 0] < partition_threshold < weight_range[weight_row, 1]):

            # If so, the spanning link is identified
            spanning_link = links[weight_row, li_idx]

            # The upstream node of the spanning link is saved (for use in the phantom link creation)
            spanning_node_upstream_name = links[weight_row, li_Mnode_u]
            try:
                spanning_node_upstream_idx = np.where(nodes[:, node_name] == spanning_node_upstream_name)[0][0]
            except:
                print(spanning_node_upstream_name, "doesn't exist in the nodes array")
                quit()

            # Return a link index
            return spanning_link

    # Otherwise return an empty string
    return spanning_link


def linear_interpolator(partition_threshold, spanning_link):
    #print("Entering linear_interpolator()")
    """
    This function is given a partition threshold and the spanning link.  It determines how far from the upstream
    node the phantom node must be placed.

    The key to this function is that the ratio of weights will be maintained in a conversion to length space. I.e.
    the weight_ratio is computed from the linear interpolation equation,

    (x - x1)/(x2 - x1) = weight_ratio   where x = partition_threshold, x1 = min weight, x2 = max weight

    Then, the length from the upstream node along the link at which the partition threshold will be found can be
    computed by multiplying the total length of the link by this weight_ratio.  This value is then returned.
    """

    # The weight_range from seek_spanning_check() is referenced
    global weight_range

    # Iterate through the links
    for link_row in range(len(links)):

        # If the link is the spanning link
        if links[link_row, li_idx] == spanning_link:

            # The total_length and start floats are saved
            total_length = float(links[link_row, lr_Length])
            start = weight_range[link_row, 0]
            break
    
    # The distance from the upstream link is computed using the linear interpolator equation
    weight_ratio = (partition_threshold - start)/weighting_function(total_length)
    length_from_start = weight_ratio*total_length

    # The distance from the upstream link to the partition_threshold is used to create a phantom node
    return length_from_start


def init_phantom_naming_convention():
    #print("Entering phantom_naming_convention()")
    """
    This function serves to solve the problem of what to call the newly created phantom nodes and links.  It determines
    the maximum number of digits that exists in the index column of the nodes and links array.  The phantom ni_idx and
    corresponding li_idx will begin at "1" starting at the next order of magnitude above the maximum.
    """
    global phantom_node_idx, phantom_link_idx

    phantom_node_idx = len(list(graph.nodes))
    phantom_link_idx = len(list(graph.edges))
    return


def phantom_node_generator(spanning_link, length_from_start, ii):
    """
    This subroutine changes the nodes and links arrays, along with the network graph, to account for the phantom node
    that has been added to the spanning link. 
    """
    global phantom_node_idx, phantom_link_idx, graph

    # Set the new integer for the phantom nodes and links
    
    phantom_node_idx = phantom_node_idx_PARAM + ii
    phantom_link_idx = phantom_link_idx_PARAM + ii

    # if i == 2:
    #     print(phantom_node_idx)
    #     print(phantom_link_idx)
    #     quit()

    # Update the nodes array with phantom idx and name
    try:
        nodes[phantom_node_idx, ni_idx] = phantom_node_idx
        nodes[phantom_node_idx, node_name] = str(phantom_node_idx)
        nodes[phantom_node_idx, ni_Image] = ii
        nodes[phantom_node_idx, ni_is_Boundary] = 1
    except IndexError:
        print("Phantom_node_idx is out of bounds again", phantom_node_idx)
    
    # Determine the nodes on the spanning link
    up_node = links[spanning_link, li_Mnode_u]
    dn_node = links[spanning_link, li_Mnode_d]

    # Determine the remaining distance of the spanning link
    remaining_length = links[spanning_link, lr_Length] - length_from_start

    # Update the links array in phantom row with idx, up_node, dn_node, length_from_start
    links[phantom_link_idx, li_idx] = phantom_link_idx
    links[phantom_link_idx, li_Mnode_u] = up_node
    links[phantom_link_idx, li_Mnode_d] = str(phantom_node_idx)
    links[phantom_link_idx, lr_Length] = length_from_start
    links[phantom_link_idx, li_Image] = ii

    # Update the links array in the spanning row with new up_node, remaining length
    links[spanning_link, li_Mnode_u] = str(phantom_node_idx)
    links[spanning_link, lr_Length] = remaining_length

    # Now dealing with the graph
    graph.add_edge(str(phantom_node_idx), up_node, weight = length_from_start)
    graph.add_edge(dn_node, str(phantom_node_idx), weight = remaining_length)
    graph.remove_edge(dn_node, up_node)
    # print(graph.edges.data('weight'))
    # print(graph.nodes)

    return

"""
Post-Processing Module

The Main module ends with a populated 3D array of the nodes that are contained within each partition.  The Post-
Processing module much first match these node-boundary definitions with the links that reside between them.

This section of the code then transforms the 3D arrays containing the nodes and links information for each partition
back into a 2D array that contains the whole system (including phantom nodes/links but excluding placeholders).  This 2D
array will be arranged such that in an automatic n-ways splitting of the array will keep the elements on one partition
together.

subnetworks_links(nodes_container):  This function populates the subnetwork_container_links array with the links
                                        "contained" within a subnetwork.  A set of endpoints (upstream and downstream
                                        nodes; each link has exactly one of each) is created for each link.  If this
                                        set is a subset of the potential_endpoints that exist within the ni_idx column
                                        of the nodes_container, then both endpoints for a link are in the subnetwork.
                                        Therefore, the link must be contained.

reorganize_arrays(nodes_container, links_container): This function consolidates the 3D nodes/links containers into 2D
                                                     arrays are arranged in terms of the partition they appear in.  If
                                                     these arrays were to be passed to Fortran CoArrays, they would be
                                                     sent to processors in alignment with the partitioning scheme.
"""

def partitioned_links(ii):
    global accounted_for_edges, links
    for edge in accounted_for_edges.edges:
        l_row = np.where(np.logical_and(links[:, li_Mnode_u] == edge[1], links[:, li_Mnode_d] == edge[0]))
        if links[l_row, li_Image] == None:
            links[l_row, li_Image] = ii
    return


def partitioned_links_old():
    global nodes, links, graph
    for link_row in range(len(links)):
        endpoint1 = links[link_row, li_Mnode_u]
        endpoint2 = links[link_row, li_Mnode_d]
        up_node = np.where(nodes[:, node_name] == endpoint1)[0][0]
        dn_node = np.where(nodes[:, node_name] == endpoint2)[0][0]
        up_image = nodes[up_node, ni_Image]
        dn_image = nodes[dn_node, ni_Image]
        links[link_row, li_Image] = dn_image
        # if up_image != dn_image:
        #     nodes[dn_node, ni_is_Boundary] = nodes[dn_node, ni_is_Boundary] + 1
    return

def check_boundary_nodes():
    global nodes, links, graph
    # print("In Check Boundary Nodes")
    for nn in nodes[:, ni_idx]:
        this_node_name = nodes[nn, node_name]
        node_adj_links = []
        node_adj_images = []
        try:
            neighbor_edges = np.where(links[:, li_Mnode_u] == this_node_name)[0]
            for link_row in neighbor_edges:
                node_adj_links.append(link_row)
        except:
            pass
        try:
            neighbor_edges = np.where(links[:, li_Mnode_d] == this_node_name)[0]
            for link_row in neighbor_edges:
                node_adj_links.append(link_row)
        except:
            pass

        for link_row in node_adj_links:
            node_adj_images.append(links[link_row, li_Image])
            nodes[nn, ni_is_Boundary] = len(set(node_adj_images)) - 1
    # print(nodes)
    # print(links)
    # quit()
    return

def connectivity_metric():
    connectivity = sum(nn for nn in nodes[:, ni_is_Boundary] if nn != None)
    return connectivity

def part_size_balance_metric(partitions):
    part_size_array = np.empty((partitions, 1), dtype=object)
    for ii in range(partitions):
        # image_link_length = 0
        # image_links = np.where(links[:, li_Image] == ii)[0]
        image_links = np.where(np.logical_and(links[:, li_Image] == ii, links[:, li_idx] != None))[0]
        # for ll in image_links:
        #     image_link_length = image_link_length + links[ll, lr_Length]
        part_size_array[ii] = sum(links[image_links, lr_Length])
    part_size_balance = (max(part_size_array) - min(part_size_array))/lr_Target
    return part_size_balance


# Initialize things that for some reason need to be globally defined
# graph = nx.DiGraph()
init_accounted_for_edges = nx.DiGraph()
# partitions = [2, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
partitions = [80, 160, 320, 640, 1280]
# partitions = [2, 3, 5, 10, 20, 40]


TRB = 'Colorado'

sprnt_file = "./TRB_SPRNT_Files/" + TRB + "_1hr_interval.spt"

# The link and node arrays are initialized and populated with the values from the csv files
f_sprnt = open(sprnt_file, "r")
sprnt_contents = f_sprnt.readlines()

init_graph = initialization()
# graph = Case2()W
# print("The graph is a tree:", is_tree)

# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     bipquick()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

# quit()
for ii in partitions:
    accounted_for_edges = init_accounted_for_edges.copy()
    graph = init_graph.copy()
    print(len(graph.nodes))
    linknode_output = "../../Output/" + TRB + "/" + TRB + "_BIPquick" + str(ii) + ".pkl"
    # path_generator(TRB, ii)
    bipquick(graph, ii)
    graph.clear()
    accounted_for_edges.clear()
