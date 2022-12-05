import numpy as np
from mpi4py import MPI


def dispatch(comm_excl, epoch, left, right, start, gap):

    """
    生成一个epoch的数据，将不同的模糊等级的数据生成放在不同的进程中
    :param comm_excl:
    :param epoch:
    :param left:
    :param right:
    :param start:
    :param gap:
    :return:
    """

    comm_rank = comm_excl.rank
    comm_size = comm_excl.size

    jobs = [(index, r) for index, r in enumerate(range(int((right - left) / gap)))]

    nums = len(jobs)

    local_jobs_offset = np.linspace(0, nums, comm_size + 1).astype('int')

    local_jobs = jobs[local_jobs_offset[comm_rank]:local_jobs_offset[comm_rank + 1]]

    print("start: rank is {}----------> {}".format(comm_rank, local_jobs))

    for cur_job in local_jobs:

        # 可以根据具体的数据生成方式生成数据
        # if os.system("python3 galaxy_noise.py --epoch {} --fwhm {} ".format(epoch, cur_job[0] * gap + start)):
        #     raise Exception("Invalid run")
        #     exit(1)

        print(cur_job[0] * gap + start)

    print("end: rank is {}----------> {}".format(comm_rank, local_jobs))
    comm_excl.Barrier()
    print("finished, enjoy!")


def start(comm_excl, epoch, left, right, start, gap):

    comm_excl.Barrier()
    dispatch(comm_excl, epoch, left, right, start, gap)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    grp = comm.Get_group()
    grp_excl = grp.Excl([0])
    comm_excl = comm.Create(grp_excl)

    print(comm_excl)

    if comm_rank > 0:
        start(comm_excl, 0, 0, 10, 1, 0.5)

    comm.Barrier()
