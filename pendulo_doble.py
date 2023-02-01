import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import matplotlib.animation as animation


def double_pendulum(y, t, L1, m1, L2, m2, g):
    
    theta_1, omega_1, theta_2, omega_2 = y
    
    n_omega_1 = -g*(2*m1+m2)*np.sin(theta_1) - m2*g*np.sin(theta_1 -2*theta_2)-2*np.sin(theta_1-theta_2)*m2*((omega_2**2)*L2 +(omega_1**2)*L1*np.cos(theta_1-theta_2))
    d_omega_1 = L1*(2*m1 + m2 -m2*np.cos(2*theta_1 - 2*theta_2))
    
    n_omega_2 = 2*np.sin(theta_1-theta_2)*((omega_1**2)*L1*(m1+m2)+g*(m1+m2)*np.cos(theta_1)+(omega_2**2)*L2*m2*np.cos(theta_1-theta_2))
    d_omega_2 = L2*(2*m1 + m2 -m2*np.cos(2*theta_1 - 2*theta_2))
    
    dydt = [omega_1, n_omega_1/d_omega_1, omega_2, n_omega_2/d_omega_2]
    return dydt

L1 = 1.0
m1 = 1.0
L2 = 1.0
m2 = 1.0
g = 9.8

# ANGULOS INCIALES Y TIEMPO TOTAL
data = [30, 50, 30]
np.savetxt('datos.txt',data)
ths = np.loadtxt('datos.txt')

y0 = np.radians([ths[0], 0.0, ths[1], 0.0])

dt = 0.01
t_stop = ths[2]
tm = (0, t_stop)

t = np.arange(0, t_stop, dt)

# SOLUCION DEL SISTEMA DE EDOS
sol = odeint(double_pendulum, y0, t, args=(L1, m1, L2, m2, g))

the1 = sol[:,0]
ome1 = sol[:,1] 
the2 = sol[:,2]
ome2 = sol[:,3] 


#############################################################
# E N E R G I A
#############################################################

def E(the1, ome1, the2, ome2):
    T = 0.5*((ome2**2)*(L2**2)*m2 + 2*ome1*ome2*L1*L2*m2*np.cos(the1-the2) + (ome1**2)*(L1**2)*(m1+m2))
    V = -(m1 + m2)*g*L1*np.cos(the1) -m2*g*L2*np.cos(the2)
    return T + V


#############################################################
# A N I M A C I O N   P E N D U L O   D O B L E
#############################################################


# CUADRO DE ANIMACION
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))

# PUNTO FIJO DEL PENDULO
circle0 = ax1.add_patch(plt.Circle([0, 0], 0.04, fc='k', zorder=3))

# POSICION INICIAL DE L1
xr10 = L1*np.sin(the1[0])
yr10 = -L1*np.cos(the1[0])
line1, = ax1.plot([0, xr10], [0, yr10], lw=2, c='k')

# POSICION INICIAL DE L1
xr20 = xr10 + L2*np.sin(the2[0])
yr20 = yr10 - L2*np.cos(the2[0])
line2, = ax1.plot([xr10, xr20], [yr10, yr20], lw=2, c='k')

# FORMA DE LA MASA 1
bob1 = 0.06
circle1 = ax1.add_patch(plt.Circle([xr10, yr10], bob1, fc='r', zorder=3))

# FORMA DE LA MASA 2
bob2 = 0.06
circle2 = ax1.add_patch(plt.Circle([xr20, yr20], bob2, fc='b', zorder=3))

# LIMITES DEL CUADRO
ax1.set_xlim(-L1*2, L1*2)
ax1.set_ylim(-L1*2.5, L1)

# PINTA LA LINEA POR LA QUE PASA LA MASA 2
history_x ,history_y = [], []
ln_h, = ax1.plot([],[],'orange')
time = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
    
def animate(i):
    # CORDENADAS DE LA MASA RESPECTO DEL TIEMPO
    xc1 = [L1*np.sin(the1[j]) for j in range(len(the1))] 
    yc1 = [-L1*np.cos(the1[j]) for j in range(len(the1))] 
    xc2 = [xc1[j] + L2*np.sin(the2[j]) for j in range(len(the1))] 
    yc2 = [yc1[j] - L2*np.cos(the2[j]) for j in range(len(the1))] 

    # CAMINO DE LA SEGUNDA MASA
    history_x.append(xc2[i])
    history_y.append(yc2[i])
    ln_h.set_data(history_x,history_y)

    # POSICION DE LAS MASAS Y CUERDAS 
    line1.set_data([0, xc1[i]], [0, yc1[i]])
    circle1.set_center((xc1[i], yc1[i]))
    line2.set_data([xc1[i], xc2[i]], [yc1[i], yc2[i]])
    circle2.set_center((xc2[i], yc2[i]))
    
    # TIEMPO
    time_text.set_text(time % (i*dt))
    
    

#############################################################
# A N I M A C I O N   T H E T A   V S   T
#############################################################
    
xdata1, ydata1 = [], []
ln1, = ax2.plot([],[],'r')
xdata2, ydata2 = [], []
ln2, = ax2.plot([],[],'b')
time_text_2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

def init():
    ax2.set_xlim(0,t_stop)
    ax2.set_ylim(-max(the2),max(the2))


def update(i):
    xdata1.append(t[i])
    ydata1.append(the1[i])
    ln1.set_data(xdata1, ydata1)
    xdata2.append(t[i])
    ydata2.append(the2[i])
    ln2.set_data(xdata2, ydata2)
    
    time_text_2.set_text(time % (i*dt))




#############################################################
# C O R R E R   A N I M A C I O N
#############################################################

fram = len(sol)
inter = dt*1000

ani_1=animation.FuncAnimation(fig, animate, frames=fram, repeat=False, interval=inter)

ani_2=animation.FuncAnimation(fig, update, frames=fram, repeat=False, interval=inter, init_func=init)


plt.subplots_adjust(bottom=0.25, wspace=0.2)


plt.show()