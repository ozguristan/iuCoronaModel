# coding=utf-8
import numpy as np


class integralci:
    def __init__(self, fler, inisillar, artimSayisi, baslangic, bitis):
        self.fler = fler
        self.inisillar = inisillar
        self.artimSayisi = artimSayisi
        self.U = np.zeros((len(inisillar), artimSayisi))
        self.U[:, 0] = inisillar
        self.baslangic = baslangic
        self.bitis = bitis
        self.dt = (bitis - baslangic) / self.artimSayisi

    def intSonuc(self):
        i = 0
        time = [self.baslangic]
        self.dt = (self.bitis - self.baslangic) / self.artimSayisi
        while i < self.artimSayisi - 1:
            self.U[:, i + 1] = self.U[:, i] + self.fler(self.U[:, i]) * self.dt
            time.append(self.baslangic + (i + 1) * self.dt)
            i += 1

        return self.U, time


class sirOzgur:
    def __init__(self, bulastirmaOrani, iyilesmeOrani):
        self.bulastirmaOrani = bulastirmaOrani
        self.iyilesmeOrani = iyilesmeOrani

    def __call__(self, u):
        S, I, _ = u
        a = np.asarray([
            -S * I * self.bulastirmaOrani, S * I * self.bulastirmaOrani - I * self.iyilesmeOrani, I * self.iyilesmeOrani
        ])
        return a



 """   ///
 from paper :
 Nowcasting and Forecasting the Spread of COVID-19 and Healthcare
Demand In Turkey, A Modelling Study
Abdullah Uçar* 1 , Şeyma Arslan* 2 , Yusuf Özdemir* 3
--------
Model : Compartemantel SEIR

Model Variables:------------------------------------------DEPENDENCIES
p1: Asymptomatic case proportion---------------------------
p2: Symptomatic case proportion----------------------------p2=1-p1
py: Symptomatic and will apply to hospital-----------------
ph: Symptomatic and will have mild disease-----------------ph=1-py
pi: will recover from hospital-----------------------------pi=1-pk
pk: will need ICU Bed--------------------------------------
pt: will recover from ICU----------------------------------pt=1-po
po: Fatality rate among ICU's according to IFR-------------po=((EO0/p2)/py)/pk
R0: Number of people contaminated by an infected
Tinc: Incubation period
Tinf: Infecious period
S: Susceptibles
E: Exposed
I: Infectious
H: Mild cases
IH: Recovered with mild symptoms
G: Infected but have not yet applied to the hospital
Y1: Applied to the hospital and will recover
Y2: Applied to the hospital and need ICU
YBU1: Still in ICU and will be required
Iybu: Recovered from ICU
YBU2: Still in ICU and will die
O: Died
    ///"""
class sirIU:
    def __init__(self, N1, N2, beta1, beta2, alfa1,alfa2,gama1,gama2,ph,sigma,pi,py,epsilon,delta,pt,po,mu,w,Q ):
        self.N1=N1
        self.N2=N2
        self.beta1 = beta1
        self.beta2 = beta2
        self.alfa1=alfa1
        self.alfa2=alfa2
        self.gama1=gama1
        self.gama2=gama2
        self.ph=ph
        self.sigma=sigma
        self.pi=pi
        self.py=py
        self.epsilon=epsilon
        self.delta=delta
        self.pt=pt
        self.po=po
        self.mu=mu
        self.w=w
        self.Q=Q


    def __call__(self, u):
        S1, S2, E1, E2, I1, I2, H, IH, G, Y1, Y2, IY, YBU1, Iybu, YBU2, O = u
        a = np.asarray([
            -S1/self.N1 * I1 * self.beta1,                  # Susceptibles Asymptomatic ----------------dS1/dt
            -S2 / self.N2 * I2 * self.beta2,                # Susceptibles Symptomatic------------------dS2/dt
            S1 / self.N1 * I1 * self.beta1- self.alfa1*E1,  # Exposed  Asymptomatic---------------------dE1/dt
            S2 / self.N2 * I2 * self.beta2- self.alfa2*E2,  # Exposed  Symptomatic----------------------dE2/dt
            self.alfa1*E1 - self.gama1 * I1,                # Infectious  Asymptomatic------------------dI1/dt
            self.alfa2*E2 - self.gama2 * I2,                # Infectious  Symptomatic-------------------dI2/dt
            self.gama1 * I1  + self.ph * self.gama2*I2-self.sigma*IH,  # Mild Cases---------------------dH/dt
            self.sigma * IH,                                # Recovered with Mild Cases ----------------dIH/dt
            self.ph * self.gama2 * I2 - self.epsilon*G,      # Infected but not yet applied to hospital-dG/dt
            self.pi * self.epsilon * G - self.delta*Y1,      # Applied to hospital and will recover-----dY1/dt
            self.delta * Y1,                                 # Recovered from hospital without ICU need-dIY/dt
            self.pk * self.epsilon * G - self.mu * Y2,       # Applied to hospital and need ICU---------dY2/dt
            self.pt* self.mu * Y2 - self.Q*YBU2,             # Still in ICU and will recover------------dYBU1/dt
            self.po * self.mu * Y2 - self.w * YBU2,          # Still in ICU and will die----------------dYBU2/dt
            self.w * YBU2                                    # Died-------------------------------------dO/dt
         ])
        return a

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    eee = sirOzgur(0.7, 0.3)

    ccc = integralci(eee, [.999, .001, 0], 2000, 1.0, 200.0)

    u, t = ccc.intSonuc()


    print("final values Susceptibles",u[0,-1])
    print("final values Infectibles",u[1,-1])
    print("final values Recovered",u[2,-1])
    plt.plot(t, u[0, :], label="Aday Hastalar")
    plt.plot(t, u[1, :], label="Bulasiklar")
    plt.plot(t, u[2, :], label="Bagisiklar")
    plt.legend()
    plt.show()
