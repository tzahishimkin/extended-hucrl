import resource
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import gc
import time

def gpu(item, use_GPU=True):
    if use_GPU:
        return item.cuda()
    else:
        return item

def test_multivariate(loc_mat, cov_mat):
    standard_normal_dist = MultivariateNormal(loc=loc_mat,
                                              covariance_matrix=cov_mat)
    action_sequence = standard_normal_dist.sample((10,))

def test_potrf(loc_mat, cov_mat):
    n = cov_mat.size(-1)
    [m.potrf(upper=False) for m in cov_mat.reshape(-1, n, n)]

def loop(fn_ref, n=10, size=10000, use_GPU=True, covariance_with_batch_dim=True, print_n=1):
    cov_mat = gpu(torch.zeros((2, 2)) + torch.eye(2), use_GPU)
    if covariance_with_batch_dim:
        cov_mat = cov_mat.unsqueeze(0).repeat(size, 1, 1)
    loc_mat = gpu(torch.zeros((size, 2)), use_GPU)

    for i in range(n):
        gc.collect()
        fn_ref(loc_mat, cov_mat)
        if i % print_n == 0:
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':

    n = 1000
    fn_ref = test_multivariate
    covariance_with_batch_dim = True
    print_n = n / 1

    use_GPU = True
    print("test_fn: ", fn_ref.__name__)
    print("n: ", n)
    print("use_GPU: ", use_GPU)
    print("covariance_with_batch_dim: ", covariance_with_batch_dim, "\n")

    start = time.time()
    loop(fn_ref=fn_ref, use_GPU=use_GPU, covariance_with_batch_dim=covariance_with_batch_dim, n=n, print_n=print_n)
    total = time.time() - start
    print(f"use_GPU:{use_GPU} - mean: {total / n}. total: {total}")

    use_GPU = False
    start = time.time()
    loop(fn_ref=fn_ref, use_GPU=use_GPU, covariance_with_batch_dim=covariance_with_batch_dim, n=n, print_n=print_n)
    total = time.time() - start
    print(f"use_GPU:{use_GPU} - mean: {total / n}. total: {total}")