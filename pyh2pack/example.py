import pyh2pack
import numpy as np

'''
   In Jupyter notebook, the outputs of `print_statistics/print_setting' might be redirected to terminals and will not be properly shown. 
   Solution to this problem is to use package 'wurlitzer'   
   Run `%load_ext wurlitzer` in Jupyeter. 
'''

N = 80000
krnl_dim = 1
coord = np.random.uniform(0, 1, size=(3, N))
krnl_param = np.array([1., -2.])
x = np.random.normal(size=(krnl_dim*N))


'''
   Test without precomputed proxy points
'''
#   build
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=3, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param)
#   matvec
y = A.h2matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 10000
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[(start_pt-1)*krnl_dim:end_pt*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()
A.print_setting()
A.clean()



'''
   Test with precomputed proxy points
'''
#   path to the file of storing proxy points
pp_fname = "./pp_tmp.dat"
#   build
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=3, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param, pp_filename=pp_fname)
#   matvec
y = A.h2matvec(x)
#   partial direct matvec
start_pt = 8000
end_pt = 10000
z = A.direct_matvec(x, start_pt, end_pt)
#   print the matvec error in the partial results
print(np.linalg.norm(y[(start_pt-1)*krnl_dim:end_pt*krnl_dim] - z) / np.linalg.norm(z))
#   statistic info of pyh2pack performance
A.print_statistic()
A.clean()


'''
   Test with matmul
'''
#   path to the file of storing proxy points
pp_fname = "./pp_tmp.dat"
#   build
A = pyh2pack.H2Mat(kernel="Quadratic_3D", krnl_dim=krnl_dim, pt_coord=coord, pt_dim=3, JIT_mode=1, rel_tol=1e-3, krnl_param=krnl_param, pp_filename=pp_fname)
#   matmul 
nvec = 10
xs = np.random.normal(size=(krnl_dim*N, nvec))
ys = A.h2matmul(xs)
#  partial direct sum
zs = []
start_pt = 1
end_pt = 1000
for i in range(nvec):
   zs.append(A.direct_matvec(xs[:,i], start_pt, end_pt))
zs = np.hstack([z[:,np.newaxis] for z in zs])
print(np.linalg.norm(ys[(start_pt-1)*krnl_dim:end_pt*krnl_dim, :] - zs, ord='fro') / np.linalg.norm(zs,  ord='fro'))

