from skfem import *
from params import *
from skfem.helpers import dot, ddot, grad, sym_grad, eye, trace, transpose
from skfem.visuals.matplotlib import plot, draw
import numpy as np
from matplotlib.animation import FuncAnimation

# Stress tensor
def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])

@BilinearForm
def nlstiffness(u, v, w):
    u_prev = w['u_prev']
    return ddot(C(sym_grad(u) + 0.5*dot(transpose(grad(u_prev)), grad(u))), sym_grad(v))

@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))

@BilinearForm
def mass(u, v, w):
    return rho*dot(u, v)

@BilinearForm
def jacobian(du, v, w):
    u_n = w['u_n']
    geom = 0.5 * (dot(transpose(grad(u_n)), grad(du)) + dot(transpose(grad(du)), grad(u_n)))
    return ddot(C(sym_grad(du) + geom), sym_grad(v))

@LinearForm
def force(v, w):
    x = w.x[0]
    f = v*0
    f[1] = -1e9*x/L
    return dot(f, v)

u_n = basis.zeros()
K = nlstiffness.assemble(basis, u_prev=u_n)
F = force.assemble(basis)

D = m.boundary_nodes()
dofs = basis.get_dofs(lambda x: x[0] == 0.)
free_dofs = basis.complement_dofs(dofs)

x = basis.zeros()

for it in range(int(1e6)):
    K = nlstiffness.assemble(basis, u_prev=u_n) 
    R = F - K @ u_n     # residual
    res_norm = np.linalg.norm(R[free_dofs])
    if it % 100:
        print(f'Residual norm: {res_norm}')
    if np.linalg.norm(res_norm)/np.linalg.norm(F) < 1e-3:
        break
    dR = jacobian.assemble(basis, u_n=u_n)
    du = solve(*condense(dR, R, D=dofs))
    u_n += 0.1*du

x = u_n

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    M = MeshQuad(np.array(m.p + 1 * x[basis.nodal_dofs]), m.t)
    ax = draw(M)
    plot(M, x[basis.nodal_dofs[1]], ax=ax)
    #show()

    Knl = nlstiffness.assemble(basis, u_prev=u_n) - stiffness.assemble(basis)
    #print(Knl @ u_n)
    K = stiffness.assemble(basis)
    x = solve(*condense(K, F, D=dofs))

    M = MeshQuad(np.array(m.p + 1 * x[basis.nodal_dofs]), m.t)
    ax = draw(M)
    plot(M, x[basis.nodal_dofs[1]], ax=ax)
    show()