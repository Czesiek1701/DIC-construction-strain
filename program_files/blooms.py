"""library created for creating, moving and plotting random blooms"""
import random
import matplotlib.pyplot as plt

class Bloom():
    def __init__(self, Xg, Yg, boxsize):
        self.box_size = boxsize                     # quare size wirth centres of circles inside
        self.Xg=Xg                                  # coord x
        self.Yg=Yg                                  # coord y
        self.num_of_circles=random.randint(3, 5)    # number of circles in one bloom
        self.circles_centres=[]                     # positions of circles in one bloom
        self.circles_radiuses=[]                    # radiuses of circles
        self.wspiprom()

    def wspiprom(self):
        for i in range(0, self.num_of_circles):
            self.X= random.randint(0, self.box_size) - int(self.box_size / 2)
            self.Y= random.randint(0, self.box_size) - int(self.box_size / 2)
            self.circles_centres.append([self.X, self.Y])
            self.promienieKol=random.randint(int(self.box_size / 4), int(self.box_size / 2))
            self.circles_radiuses.append(self.promienieKol)

    def plotgggg(self,j):
        return plt.Circle((self.circles_centres[j][0] + self.Xg, self.circles_centres[j][1] + self.Yg), self.circles_radiuses[j], color='black')

