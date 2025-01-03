from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres, spilu, LinearOperator
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import os
import psutil 
import time

# Parâmetros para o problema
L = 75.0  # Comprimento total do domínio (-25 a 50)
T = 45.0  # Tempo total de simulação
nx = 100  # Número de pontos espaciais
nt = 400  # Número de passos no tempo
D = 1.0   # Coeficiente de difusão
v = 0.   # Velocidade de advecção
r = 1.0   # Taxa de reação
theta = 1.0  # parâmetro entre euler implícito, crank nicolson ou euler explícito
metodo = 1  # Escolha do método (1: linalg.solve, 2: gmres, 3: gmres com precondicionamento LU)

# Função para resolver o sistema não-linear completo
def resolver_sistema_nao_linear(F, J, u_guess, max_iter=50, tol=1e-6, metodo=1):

    for iter in range(max_iter):
        f = F(u_guess)
        jac = J(u_guess)
        if metodo == 1:
            delta_u = np.linalg.solve(jac, -f)
            #numero_de_condicao = npl.cond(jac)
            #print(f"Número de Condição: {numero_de_condicao}")
        elif metodo == 2:
            jac = J(u_guess)  # Obter a matriz Jacobiana
            #numero_de_condicao = npl.cond(jac)
            #print(f"Número de Condição: {numero_de_condicao}")
            delta_u, info = gmres(jac, -f, rtol=tol)
            if info != 0:
              print("GMRES não converge")
        elif metodo == 3:
            jac_sparse = csc_matrix(jac)  # Converter para formato esparso
            precond = spilu(jac_sparse)
            # Definindo M como um operador linear
            M = LinearOperator(jac_sparse.shape, matvec=precond.solve, rmatvec=precond.solve)
            #numero_de_condicao = npl.cond(jac)
            #print(f"Número de Condição: {numero_de_condicao}")
            delta_u, info = gmres(jac, -f, rtol=tol, M=M)
            if info != 0:
             print("GMRES com precondicionamento LU não converge")

        else:
            raise ValueError("Método inválido. Escolha 1, 2 ou 3.")

        u_guess += delta_u
        #print(f"Iteração {iter + 1}: Resíduo = {np.linalg.norm(f, 2)}")  # Imprimir resíduo
        if np.linalg.norm(delta_u, 2) < tol:
            #print(f"Convergiu em {iter + 1} iterações.")  # Imprimir número de iterações
            return u_guess

    raise ValueError("Método de Newton não convergiu.")

# Função principal para resolver a equação de Fisher-KPP
def fisher_kpp(theta, L, T, nx, nt, D, v, r, u0, metodo=1):

    dx = (50 - (-25)) / (nx - 1)
    dt = T / nt

    x = np.linspace(-25, 50, nx)  # Ajuste no domínio para contemplar a condição inicial
    u = np.zeros((nx, nt + 1))
    u[:, 0] = u0(x)

    alpha = D * dt / dx**2
    beta = v * dt / (2 * dx)

    def F(u_new, u_old):
        """Define o resíduo F(u) para o sistema não-linear."""
        u_diffusion = alpha * (np.roll(u_new, -1) - 2 * u_new + np.roll(u_new, 1))
        u_advection = beta * (np.roll(u_new, -1) - np.roll(u_new, 1))
        u_reaction = r * u_new * (1 - u_new)
        u_diffusion[-1] = 0  # Condição de Neumann na fronteira direita
        u_advection[-1] = 0
        return u_new - u_old - dt * (theta * (u_diffusion - u_advection + u_reaction) + (1 - theta) * (alpha * (np.roll(u_old, -1) - 2 * u_old + np.roll(u_old, 1)) - beta * (np.roll(u_old, -1) - np.roll(u_old, 1)) + r * u_old * (1 - u_old)))

    def J(u_new):
        """Define a Jacobiana J(u) do sistema não-linear."""
        diag = 1 + dt * theta * (-2 * alpha + r * (1 - 2 * u_new))
        off_diag = np.full_like(u_new, dt * theta * alpha)
        jac = np.diag(diag) + np.diag(off_diag[:-1], k=1) + np.diag(off_diag[:-1], k=-1)
        jac[0, :] = 0
        jac[0, 0] = 1
        jac[-1, :] = 0
        jac[-1, -2] = -alpha
        jac[-1, -1] = 1 + alpha
        return jac

    for n in range(nt):
        u[:, n + 1] = resolver_sistema_nao_linear(lambda u_new: F(u_new, u[:, n]), J, u[:, n].copy(), metodo=metodo)
        u[0, n + 1] = 1  # Condição de Dirichlet na fronteira esquerda
        #norma_u = np.linalg.norm(u[:, n + 1])
        #print(f"Passo de tempo {n + 1}: Norma da solução = {norma_u}")

    return x, u, J


# Condições iniciais
def condicao_inicial(x):
    return np.where(x < -10, 1, np.where((x >= 10) & (x <= 20), 0.25, 0))

# Registrar o tempo inicial e memória inicial
start_time = time.time()
process = psutil.Process(os.getpid())
start_memory = process.memory_info().rss  # em bytes

# Resolver o problema
x, u, J_func = fisher_kpp(theta, L, T, nx, nt, D, v, r, condicao_inicial, metodo=metodo)

#print(u)

# plt.spy(J_func(u[:, 0]))  # Visualizar a esparsidade da Jacobiana no primeiro passo de tempo
# plt.show()
# print(J_func(u[:, 0]))

# Plotar os resultados
plt.figure(figsize=(10, 6))
for i in range(0, nt+1, nt // 10):
    plt.plot(x, u[:, i], label=f"t = {i * T / nt:.2f}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Solução Fisher-KPP")
plt.legend()
plt.grid()
plt.show()

# gráfico da função completa
T,X = np.meshgrid(np.linspace(0, T, nt+1), x)
Z = np.array(u)

# Create 3D plot
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, .333])
ax.plot_surface(X, T, Z)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(t, x)')
# Configurando a superfície com uma colormap roxa e contornos
surface = ax.plot_surface(X, T, Z, cmap='plasma', edgecolor='k', rstride=1000, cstride=30, linewidth=0.5)
fig.colorbar(surface, shrink=0.6, aspect=10, label="u(t, x)")
plt.show()

# # Registrar o tempo final e memória final
# end_time = time.time()
# end_memory = process.memory_info().rss  # em bytes

# # Calcular e exibir o tempo de execução e memória usada
# execution_time = end_time - start_time
# memory_used = (end_memory - start_memory) / 1024**2  # em MB
# print(f"Tempo de execução: {execution_time:.2f} segundos")
# print(f"Memória usada: {memory_used:.2f} MB")