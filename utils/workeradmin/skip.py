def skip_train(communicator, comm_size, yes=False):
    """Broadcast out to workers if skipping training section of pipeline.
    Send 1 if skipping training stage; send 0 if executing.
    
    Keyword arguments:
    communicator -- comm variable used to communicate with nodes (typically comm = MPI.COMM_WORLD).
    comm_size -- how many nodes exist in the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).
    yes -- skip or do not skip training stage. True: skip training stage. False: do not skip training stage (default: False)."""
    node_rank = [i+1 for i in range(comm_size-1)]
    if yes is True:
        msg = [1**1 for j in range(comm_size-1)]
        node_iter = 0
        for skip in msg:
            communicator.send(skip, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter +=1

    else:
        msg = [0**1 for j in range(comm_size-1)]
        node_iter = 0
        for no_skip in msg:
            communicator.send(no_skip, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter += 1


def skip_attack(communicator, comm_size, yes=False):
    """Broadcast out to workers if skipping attack section of pipeline.
    Send 1 if skipping training stage; send 0 if executing.
    
    Keyword arguments:
    communicator -- comm variable used to communicate with nodes (typically comm = MPI.COMM_WORLD).
    comm_size -- how many nodes exist in the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).
    yes -- skip or do not skip attack stage. True: skip attack stage. False: do not skip attack stage (default: False)."""
    node_rank = [i+1 for i in range(comm_size-1)]
    if yes is True:
        msg = [1**1 for j in range(comm_size-1)]
        node_iter = 0
        for skip in msg:
            communicator.send(skip, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter +=1

    else:
        msg = [0**1 for j in range(comm_size-1)]
        node_iter = 0
        for no_skip in msg:
            communicator.send(no_skip, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter += 1


# TODO: Write out once cleaning stage is added
# def train(communicator, comm_size, yes=False):
#     pass
