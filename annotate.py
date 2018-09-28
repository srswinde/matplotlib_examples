import pandas as pd
import math
from astro import locales;loc=locales.mtlemmon()
from astro.angles import Deg10, Dec_angle, Hour_angle, Angle

# At Dec = +46, HA = 00:35, the dome diff was -5.3 deg just
#     before it moved and +1.2 deg right after it stopped moving
#  At Dec = +46, HA = 02:55, dome diff was -2.55 deg just before
#     it moved and +0.65 deg after
#  At Dec = +29, HA = 02:45, dome diff was -2.6 deg just before
#     it moved and +0.6 deg after
#The latter, at Dec +29, shows it doesn't matter whether the dome is
#moving clockwise or counterclockwise - same asymmetry.

import matplotlib.pyplot as plt
import numpy as np
def makedata(ndps=90, w=1.5):
    df = pd.DataFrame(index=range(ndps), columns=["elevation", "width"])
    for a in range(ndps):
        elevation = a*90/ndps
        df.iloc[a].elevation = elevation
        cosel = math.cos(elevation*math.pi/180.0)
        df.iloc[a].width = w/math.cos(elevation*math.pi/180.0)

    df.index=df.elevation

    return df['width']



def mkplt(df):
    fig=plt.gcf()
    ax=fig.add_subplot(111, projection='polar')
    ax.set_yticks(range(0, 90, 10))
    ax.set_yticklabels(map(str, range(90, 0, -10)))
    ax.set_xlim(0, math.pi)
    for idx in range(len(df)):
        aw=df.iloc[idx]
        R = 90-df.index[idx]
        angs=np.linspace(0, aw, 50)*math.pi/180
        ax.plot(angs-(aw/2)*math.pi/180, [R]*50 )

    plt.annotate()

def mkbarplt(df):
    df.plot.bar()
    ax = plt.gca()
    coords, hadecs = betsy()
    for x,y in coords:
        ax.plot([x.deg10], 10, 'rx')

    ax.grid()
    ax.set_ylim(0,30)
    ax.set_ylabel("Dome Error Window in degrees")
    ax.set_xlabel("Telescope Elevation")
    ax.set_xticks(range(0,90, 10))
    ax.set_xticklabels(range(0,90, 10))
    for idx,(el, az) in enumerate(coords):
        if idx==1:
            va='top'
            offx=20
            offy=-20
        else:
            va="bottom"
            offx=-20
            offy=20
            plt.annotate("{} {} el={:4.1f}".format( str(hadecs[idx][0]), str(hadecs[idx][1]), el.deg10 ),
                     xy=(el.deg10,10), xytext=(offx, offy),
                     textcoords='offset points', ha='right', va=va,
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))




def mapr(r):
    return 90-r


def betsy():
    coords=[]

    hadecs = [(Hour_angle([0,35,0]), Dec_angle([46,0,0])),
              (Hour_angle([2,55,0]), Dec_angle([46,0,0])),
              (Hour_angle([2,45,0]), Dec_angle([29,0,0]))]
    for ha, dec in hadecs:
        coords.append(loc.hadec2hor(ha, dec))

    return coords, hadecs

