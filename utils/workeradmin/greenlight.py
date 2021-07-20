def killmsg(communicator, comm_size: int, yes=False) -> None:
    """
    Kill or Greenlight all workers. Send 1 to greenlight workers to continue execution.
    Send 0 to kill all workers in the MPI.COMM_WORLD. Prevents blocking if terminating Jespipe.
    
    ### Parameters:
    :param communicator: Communicator variable used to communicate with nodes in the 
    MPI.COMM_WORLD (typically comm = MPI.COMM_WORLD).
    :param comm_size: Size of the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).
    :param yes: Kill or do not kill all worker nodes. True: kill all workers. False: do not kill all workers (default: False).
    """
    node_rank = [i+1 for i in range(comm_size-1)]
    if yes is True:
        msg = [0**1 for j in range(comm_size-1)]
        node_iter = 0
        for death in msg:
            communicator.send(death, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter +=1

    else:
        msg = [1**1 for j in range(comm_size-1)]
        node_iter = 0
        for life in msg:
            communicator.send(life, dest=node_rank[node_iter], tag=node_rank[node_iter])
            node_iter += 1
