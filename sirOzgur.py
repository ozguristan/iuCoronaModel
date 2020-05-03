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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    eee = sirOzgur(0.7, 0.3)

    ccc = integralci(eee, [.999, .001, 0], 200, 1.0, 200.0)

    u, t = ccc.intSonuc()


    print("final values Susceptibles",u[0,-1])
    print("final values Infectibles",u[1,-1])
    print("final values Recovered",u[2,-1])
    plt.plot(t, u[0, :], label="Susceptible")
    plt.plot(t, u[1, :], label="Infected")
    plt.plot(t, u[2, :], label="Recovered")
    plt.legend()
    plt.show()
