import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc

rc['font.family'] = 'serif'
rc['font.size'] = 14
madcmap = 'plasma'
fw = 5; fl = 4
lw = 3

# Carbonfox colors
black = "#282828"
red = "#ee5396"
green = "#25be6a"
cyan = "#08bdba"
purple = "#78a9ff"
mauve = "#be95ff"
blue = "#33b1ff"
gray = "#dfdfe0"

def sec_to_msg(tics:float):
    '''
    Take value representing duration in seconds and convert to string of form "Xh Xm X.Xs" if above an hour, "Xm X.Xs" if above a minute, and "X.Xs" otherwise.
    '''
    return f"{tics:.1f}s" if tics<60 else (f"{tics//60:.0f}m {tics%60:.1f}s" if tics<3600 else f"{tics//3600:.0f}h {(tics%3600)//60:.0f}m {tics%60:.1f}s")

def rgb_to_greenblind(clr):
    '''
    Convert input 3-tuple of RGB into red-green colorblind version for checking visibility.
    '''
    clr = clr * 255
    new_clr = np.zeros(3)
    new_clr[0] = (4211 + .677*clr[1]**2.2 + 0.2802*clr[0]**2.2)**(1/2.2)
    new_clr[1] = (4211 + .677*clr[1]**2.2 + 0.2802*clr[0]**2.2)**(1/2.2)
    new_clr[2] = (4211 + .95724*clr[2]**2.2 + 0.02138*clr[1]**2.2 - 0.02138*clr[0]**2.2)**(1/2.2)
    return new_clr/255

def interp_colors(clr_min,clr_max,num_total:int=5):
    '''
    Given two 3-tuples of RGB values, return list of colors interpolating between them.
    '''
    if num_total>2:
        alph = np.linspace(0,1,num_total)[1:-1]
        clrs = (
            [clr_min]
            + list( clr_min[None] * ( 1-alph[:,None] ) + clr_max[None] * alph[:,None] )
            + [clr_max]
        )
    else:
        clrs = [clr_min, clr_max]
    return clrs

def madimshow(mat,cmap:str=madcmap,xlabel:str='',ylabel:str='',axis=True,figsize=(4,4),vmin=None,vmax=None):
    '''
    Convenient function for pyplot imshow with personal preferred defaults. (primarily for checking)
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    args = {'cmap':cmap}
    if vmin is not None:
        args['vmin'] = vmin
    if vmax is not None:
        args['vmax'] = vmax
    ax.imshow(mat,**args)
    if len(xlabel)>0:
        ax.set_xlabel(xlabel)
    if len(ylabel)>0:
        ax.set_ylabel(ylabel)
    if not axis:
        ax.axis('off')
    fig.tight_layout()

def boxplot_args(clr:str='k', lw:float=2, alpha:float=.7, sz:float=8, mkr:str='o'):
    '''
    Return dictionary of boxplot arguments with quick customization.
    Usage:

    args = boxplot_args('red')
    plt.boxplot(data, **args)
    '''
    args = {
        "boxprops": dict(color=clr, linewidth=lw, alpha=alpha),
        "whiskerprops": dict(color=clr, linewidth=lw, alpha=alpha),
        "capprops": dict(color=clr, linewidth=lw, alpha=alpha),
        "medianprops": dict(color=clr, linewidth=lw, alpha=alpha),
        "flierprops": dict(markerfacecolor='w',markeredgecolor=clr, markersize=sz, marker=mkr, alpha=1.),
    }
    return args

def get_gridplot(fw:float=5,fl:float=4,grid:bool=True):
    '''
    Convenient initialization of figure.
    '''
    fig,ax = plt.subplots(figsize=(fw,fl))
    ax.grid(grid)
    return fig,ax

def get_gridplots(rows:int,cols:int,fw:float=5,fl:float=4,grid:bool=True):
    '''
    Convenient initialization of figure with multiple subplots.
    '''
    fig,axes = plt.subplots(rows,cols,figsize=(fw*cols,fl*rows))
    if (rows==1 and cols>1) or (rows>1 and cols==1):
        _ = [ax.grid(grid) for ax in axes]
    elif (rows>1 and cols>1):
        _ = [axes[i][j].grid(grid) for i in range(rows) for j in range(cols)]
    else:
        axes.grid(grid)
    return fig,axes

def get_marker_args(c='k',m='o',sz:float=8,
    zo=None, alph:float=1.):
    '''
    Return dictionary of plot arguments (markers only) with quick customization.
    Usage:

    args = get_marker_args('red','x')
    plt.plot(x,y, **args)
    '''
    args = { "c":c, "linestyle":'', "marker":m, "markersize":sz, "alpha":alph }
    if zo is not None:
        args = args | {"zorder":zo}
    return args

def get_line_args(c='k',ls='-',lw:float=3,
    zo=None, alph:float=1.):
    '''
    Return dictionary of plot arguments (line only) with quick customization.
    Usage:

    args = get_line_args('red','--')
    plt.plot(x,y, **args)
    '''
    args = { "c":c, "linewidth":lw, "linestyle":ls, "alpha":alph }
    if zo is not None:
        args = args | {"zorder":zo}
    return args

def get_mrk_ln_args(c='k',m='o',ls='-',sz:float=8,lw:float=3,
    zo=None, alph:float=1.):
    '''
    Return dictionary of plot arguments (markers and line) with quick customization.
    Usage:

    args = get_mrk_ln_args('red','o','--')
    plt.plot(x,y, **args)
    '''
    args = { "c":c, "linewidth":lw, "linestyle":ls, "marker":m, "markersize":sz, "alpha":alph }
    if zo is not None:
        args = args | {"zorder":zo}
    return args



# ---------------------------------------
# COLORBLIND-FRIENDLY PALETTES

# Bright qualitative
bq = {
    'blue': np.array([68,119,170])/255,
    'cyan': np.array([102,204,238])/255,
    'green': np.array([34,136,51])/255,
    'yellow': np.array([204,187,68])/255,
    'red': np.array([238,102,119])/255,
    'purple': np.array([170,51,119])/255,
    'gray': np.array([187,187,187])/255
}

# High-contrast qualitative
hcq = {
    'yellow': np.array([221,170,51])/255,
    'red': np.array([187,85,102])/255,
    'blue': np.array([0,68,136])/255
}

# Vibrant qualitative
vq = {
    'blue': np.array([0,119,187])/255,
    'cyan': np.array([51,187,238])/255,
    'teal': np.array([0,153,136])/255,
    'orange': np.array([238,119,51])/255,
    'red': np.array([204,51,17])/255,
    'magenta': np.array([238,51,119])/255,
    'gray': np.array([187,187,187])/255
}

# Muted qualitative
mq = {
    'indigo': np.array([51,34,136])/255,
    'cyan': np.array([136,204,238])/255,
    'teal': np.array([68,170,153])/255,
    'green': np.array([17,119,51])/255,
    'olive': np.array([153,153,51])/255,
    'sand': np.array([221,204,119])/255,
    'rose': np.array([204,102,119])/255,
    'wine': np.array([136,34,85])/255,
    'purple': np.array([170,68,153])/255,
    'gray': np.array([221,221,221])/255,
}

# Medium-contrast qualitative
mcq = {
    'light_yellow': np.array([238,204,102])/255,
    'light_red': np.array([238,153,170])/255,
    'light_blue': np.array([102,153,204])/255,
    'dark_yellow': np.array([153,119,0])/255,
    'dark_red': np.array([153,68,85])/255,
    'dark_blue': np.array([0,68,136])/255
}

# Pale and dark qualitative
pq = {
    'pale_blue': np.array([187,204,238])/255,
    'pale_cyan': np.array([204,238,255])/255,
    'pale_green': np.array([204,221,170])/255,
    'pale_yellow': np.array([238,238,187])/255,
    'pale_red': np.array([255,204,204])/255,
    'pale_gray': np.array([221,221,221])/255,
    'dark_blue': np.array([34,34,85])/255,
    'dark_cyan': np.array([34,85,85])/255,
    'dark_green': np.array([34,85,34])/255,
    'dark_yellow': np.array([102,102,51])/255,
    'dark_red': np.array([102,51,51])/255,
    'dark_gray': np.array([85,85,85])/255
}

# Light qualitative
lq = {
    'blue': np.array([119,170,221])/255,
    'cyan': np.array([153,221,255])/255,
    'mint': np.array([68,187,153])/255,
    'pear': np.array([187,204,51])/255,
    'olive': np.array([170,170,0])/255,
    'yellow': np.array([238,221,136])/255,
    'orange': np.array([238,136,102])/255,
    'pink': np.array([255,170,187])/255,
    'gray': np.array([221,221,221])/255
}

# Sunset diverging
sunset = [
    np.array([ 54,75,154 ])/255,
    np.array([ 74,123,183 ])/255,
    np.array([ 110,166,205 ])/255,
    np.array([ 152,202,225 ])/255,
    np.array([ 194,228,239 ])/255,
    np.array([ 234,236,204 ])/255,
    np.array([ 254,218,139 ])/255,
    np.array([ 253,179,102 ])/255,
    np.array([ 246,126,75 ])/255,
    np.array([ 221,61,45 ])/255,
    np.array([ 165,0,38 ])/255
]

# Nightfall diverging
nightfall = [
    np.array([ 18,90,86 ])/255,
    np.array([ 0,118,123 ])/255,
    np.array([ 35,143,157 ])/255,
    np.array([ 66,167,198 ])/255,
    np.array([ 96,188,233 ])/255,
    np.array([ 157,204,239 ])/255,
    np.array([ 198,219,237 ])/255,
    np.array([ 222,230,231 ])/255,
    np.array([ 236,234,218 ])/255,
    np.array([ 240,230,178 ])/255,
    np.array([ 249,213,118 ])/255,
    np.array([ 255,185,84 ])/255,
    np.array([ 253,154,68 ])/255,
    np.array([ 245,118,52 ])/255,
    np.array([ 233,76,31 ])/255,
    np.array([ 209,24,7 ])/255,
    np.array([ 160,24,19 ])/255
]

# BuRd diverging
burd = [
    np.array([ 33,102,172 ])/255,
    np.array([ 67,147,195 ])/255,
    np.array([ 146,197,222 ])/255,
    np.array([ 209,229,240 ])/255,
    np.array([ 247,247,247 ])/255,
    np.array([ 253,219,199 ])/255,
    np.array([ 244,165,130 ])/255,
    np.array([ 214,96,77 ])/255,
    np.array([ 178,24,43 ])/255
]

# PRGn diverging
prgn = [
    np.array([ 118,42,131 ])/255,
    np.array([ 153,112,171 ])/255,
    np.array([ 194,165,207 ])/255,
    np.array([ 231,212,232 ])/255,
    np.array([ 247,247,247 ])/255,
    np.array([ 217,240,211 ])/255,
    np.array([ 172,211,158 ])/255,
    np.array([ 90,174,97 ])/255,
    np.array([ 27,120,55 ])/255
]

# YlOrBr sequential
ylorbr = [
    np.array([ 255,255,229 ])/255,
    np.array([ 255,247,188 ])/255,
    np.array([ 254,227,145 ])/255,
    np.array([ 254,196,79 ])/255,
    np.array([ 251,154,41 ])/255,
    np.array([ 236,112,20 ])/255,
    np.array([ 204,76,2 ])/255,
    np.array([ 153,52,4 ])/255,
    np.array([ 102,37,6 ])/255
]

# Iridescent sequential
iridescent = [
    np.array([ 254,251,233 ])/255,
    np.array([ 252,247,213 ])/255,
    np.array([ 245,243,193 ])/255,
    np.array([ 234,240,181 ])/255,
    np.array([ 221,236,191 ])/255,
    np.array([ 208,231,202 ])/255,
    np.array([ 194,227,210 ])/255,
    np.array([ 181,221,216 ])/255,
    np.array([ 168,216,220 ])/255,
    np.array([ 155,210,225 ])/255,
    np.array([ 141,203,228 ])/255,
    np.array([ 129,196,231 ])/255,
    np.array([ 123,188,231 ])/255,
    np.array([ 126,178,228 ])/255,
    np.array([ 136,165,221 ])/255,
    np.array([ 147,152,210 ])/255,
    np.array([ 155,138,196 ])/255,
    np.array([ 157,125,178 ])/255,
    np.array([ 154,112,158 ])/255,
    np.array([ 144,99,136 ])/255,
    np.array([ 128,87,112 ])/255,
    np.array([ 104,73,87 ])/255,
    np.array([ 70,53,58 ])/255
]

# Incandescent sequential
incandescent = [
    np.array([ 206,255,255 ])/255,
    np.array([ 198,247,214 ])/255,
    np.array([ 162,244,155 ])/255,
    np.array([ 187,228,83 ])/255,
    np.array([ 213,206,4 ])/255,
    np.array([ 231,181,3 ])/255,
    np.array([ 241,153,3 ])/255,
    np.array([ 246,121,11 ])/255,
    np.array([ 249,73,2 ])/255,
    np.array([ 228,5,21 ])/255,
    np.array([ 168,0,3 ])/255
]






ibm_reds = [
    "#760039",
    "#980D50",
    "#BA1968",
    "#DC267F",
    "#E865A5",
    "#F3A3CA",
    "#FFE2F0",
]

ibm_blues = [
    "#0A349F",
    "#2852BF",
    "#4671DF",
    "#648FFF",
    "#8FAEFF",
    "#BACDFF",
    "#E5ECFF",
]

ibm_purples = [
    "#341E97",
    "#4B33B5",
    "#6149D2",
    "#785ef0",
    "#9783F5",
    "#B7A8FA",
    "#D6CDFF",
]

ibm_oranges = [
    "#A23E00",
    "#C14A00",
    "#DF5500",
    "#FE6100",
    "#FE8B43",
    "#FFB486",
    "#FFDEC9",
]

ibm_yellows = [
    "#A57100",
    "#C38600",
    "#E19B00",
    "#ffb000",
    "#FFC647",
    "#FFDC8F",
    "#FFF2D6",
]

ibm_red = ibm_reds[3]
ibm_blue = ibm_blues[3]
ibm_purple = ibm_purples[3]
ibm_orange = ibm_oranges[3]
ibm_yellow = ibm_yellows[3]