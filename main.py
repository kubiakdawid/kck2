#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib
matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import colorsys
#można powyższego używać

from matplotlib import colors

def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    #rc('text', usetex=True) 
    #rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)


    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')

def hsv2rgb(h,s,v):
    if s==0:
        return (v,v,v)
    h_i = int(h*6)
    f = (h*6)-h_i
    p = v*(1-s)
    q = v*(1-(s*f))
    t = v*(1-(s*(1-f)))
    h_i = h_i%6
    if h_i==0: return (v,t,p)
    if h_i==1: return (q,v,p)
    if h_i==2: return (p,v,t)
    if h_i==3: return (p,q,v)
    if h_i==4: return (t,p,v)
    if h_i==5: return (v,p,q)
    return (0,0,0)

def gradient_rgb_bw(v):
    return (v,v,v)

def gradient_rgb_gbr(v):
    if (v<=0.5):
        return (0,1-2*v,2*v)
    else:
        v -= 0.5
        return (2*v,0,1-2*v)

def gradient_rgb_gbr_full(v):
    if v<=0.25:
        return (0,1,v*4)
    elif v<=0.5:
        return (0,1-(v-0.25)*4,1)
    elif v<=0.75:
        return ((v-0.5)*4,0,1)
    else:
        return (1,0,1-(v-0.75)*4)

def gradient_rgb_wb_custom(v):
    if v<=1/7:
        return (1,1-(v*7),1)
    elif v<=2/7:
        return (1-(v-1/7)*7,0,1)
    elif v<=3/7:
        return (0,(v-2/7)*7,1)
    elif v<=4/7:
        return (0,1,1-(v-3/7)*7)
    elif v<=5/7:
        return ((v-4/7)*7,1,0)
    elif v<=6/7:
        return (1,1-(v-5/7)*7,0)
    else:
        return (1-(v-6/7)*7,0,0)

def gradient_hsv_bw(v):
    return hsv2rgb(0,0,v)

def gradient_hsv_gbr(v):
    h = (1.0/3.0)+v*(2.0/3.0)
    return hsv2rgb(h,1,1)

def gradient_hsv_unknown(v):
    h = (1.0/3.0)*(1-v)
    return hsv2rgb(h,0.5,1)

def gradient_hsv_custom(v):
    return hsv2rgb(v,1-v*v,1)


def mapa():
    f = open('big.dem', 'r')
    linia = f.readline().split()
    szer, wys = int(float(linia[0])), int(float(linia[1]))
    dane = np.zeros((wys, szer))
    for i in range(wys):
        dane[i, :] = [float(x) for x in f.readline().split()]
    f.close()

    mn,mx = np.min(dane),np.max(dane)
    norm=(dane - mn)/(mx - mn)
    h = (1.0-norm)*0.33

    cien = np.zeros((wys, szer))
    cien[:,1:] = dane[:,1:]-dane[:,:-1]
    v = 0.8+cien*0.02
    v[v>1] =1
    v[v<0] = 0

    rgb = np.zeros((wys, szer,3))
    hi =(h*6).astype(int)%6
    f = (h*6)- hi
    p =v*0
    q = v*(1-f)
    t = v*(1-(1-f))

    rgb[hi==0,0],rgb[hi ==0,1],rgb[hi == 0,2]= v[hi ==0],t[hi ==0],p[hi==0]
    rgb[hi==1,0],rgb[hi==1,1],rgb[hi ==1,2] =q[hi ==1], v[hi ==1], p[hi ==1]
    rgb[hi==2,0],rgb[hi==2,1],rgb[hi ==2,2]=p[hi ==2],v[hi==2],t[hi==2]
    rgb[hi==3,0],rgb[hi==3,1],rgb[hi== 3,2]=p[hi ==3],q[hi==3],v[hi==3]
    rgb[hi==4,0],rgb[hi==4,1],rgb[hi== 4,2]=t[hi ==4],p[hi==4],v[hi==4]
    rgb[hi==5,0],rgb[hi==5,1],rgb[hi== 5,2] =v[hi== 5], p[hi==5],q[hi==5]

    plt.figure(figsize=(10,10))
    plt.imshow(rgb)
    plt.savefig('mapa.pdf')


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])

    mapa()
