from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""Bare bones skeletal structure of what the MPI application will
   look like. Will flesh out more as more pieces of the pipeline
   are completed. **Will use 8 nodes on Ursula but I don't want
   to kill my computer during my initial development**"""
if rank == 0:
    data1 = {"a": 42, "b": 69}
    data2 = {"c": 3.14, "d": 0.5}
    data3 = {"e": 2, "f": 100}
    
    # Send out data
    comm.send(data1, dest=1, tag=10)
    comm.send(data2, dest=2, tag=11)
    comm.send(data3, dest=3, tag=12)

    # Receive report back from workers
    comm.recv(source=1, tag=14)
    comm.recv(source=2, tag=15)
    comm.recv(source=3, tag=16)

elif rank == 1:
    data = comm.recv(source=0, tag=10)
    print(data)
    print("Hello from worker #1!")
    comm.send("", dest=0, tag=14)

elif rank == 2:
    data = comm.recv(source=0, tag=11)
    print(data)
    print("Hello from worker #2!")
    comm.send("", dest=0, tag=15)

elif rank == 3:
    data = comm.recv(source=0, tag=12)
    print(data)
    print("Hello from worker #3!")
    comm.send("", dest=0, tag=16)
