'''
Created 19 Jun 2023
Editted 
thompson@knmi.nl

Functions for loading dataensemble boosting data (ETHZ)
- data from '/net/pc200023/nobackup/users/thompson/ETHZ'
>conda activate butterfly 
'''

# Load neccessary libraries
import subprocess
import iris
import iris.coord_categorisation as icc
from iris.coord_categorisation import add_season_membership
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy as cart
import glob
import matplotlib.cm as mpl_cm
import sys
import scipy.stats as sps
from scipy.stats import genextreme as gev
import random
import scipy.io
import xarray as xr
import netCDF4 as nc
import iris.coords
import iris.util
from iris.util import equalise_attributes
from iris.util import unify_time_units
from scipy.stats.stats import pearsonr
import calendar
import cartopy.feature as cf


def ETHZ_data(var, region):
    '''
    var: 'PSL', 'Z500', 'pr_mm',
    '''
    filepath = '/net/pc200023/nobackup/users/thompson/ETHZ'
    files = sorted(glob.glob(filepath+'/'+var+'*'))
    latConstraint = iris.Constraint(latitude=lambda cell: region[1] < cell < region[0])
    lonConstraint = iris.Constraint(longitude=lambda cell: region[3] < cell < region[2])
    cube_list = iris.load(files).extract(latConstraint)
    # make -180 to 180
    new_cube_list = iris.cube.CubeList([])
    for each in cube_list:
        new_cube_list.append(roll_cube_180(each))
    # extract JJA
    jja_list = iris.cube.CubeList([])
    for each in new_cube_list:
        #iris.util.promote_aux_coord_to_dim_coord(each, 'time')
        each = each.extract(lonConstraint)
        iris.coord_categorisation.add_year(each, 'time')
        iris.coord_categorisation.add_month(each, 'time')
        iris.coord_categorisation.add_day_of_month(each, 'time')
        iris.coord_categorisation.add_season(each, 'time')
        jja_list.append(each.extract(iris.Constraint(season='jja')))
    return jja_list

def ETHZ_format(var, cube_list):
    filepath = '/net/pc200023/nobackup/users/thompson/ETHZ'
    files = sorted(glob.glob(filepath+'/'+var+'*'))
    # add realisation coord
    for n, each in enumerate(cube_list):
        real = files[n].split('ssp370_r')[1].split('i')[0]
        new_coord = iris.coords.AuxCoord(int(real), 'realization')
        each.add_aux_coord(new_coord)
    # merge each realisation
    for each in cube_list:
        iris.util.promote_aux_coord_to_dim_coord(each, 'time')
    equalise_attributes(cube_list)
    unify_time_units(cube_list)
    cube_list = cube_list.concatenate() # should have 1 cube per realization
    #cube_list = cube_list.merge_cube() # not all same lengt so doesnt work
    return cube_list

def ETHZ_boosted_data(var, date):
    '''
    Loads cube of specified date and variable
    
    date: YYYYMMDD, string, options 2007-08-07; 2009-08-08; 2016-07-12
    var: string, from in the dictionary 'dict_var'. Key ones: TAS, Z500, CP, LP, 
    '''
    dict_var = {'TAS': 'Reference height temperature', 'SLP': 'Sea level pressure', 'CP': 'Convective precipitation rate (liq + ice)', 'LP': 'Large-scale (stable) precipitation rate (liq + ice)', 'Z500': 'Geopotential Z at 500 mbar pressure surface', 'HUM': 'Reference height humidity', 'Z200': 'Geopotential Z at 200 mbar pressure surface', 'V200': 'Meridional wind at 200 mbar pressure surface', 'Z700': 'Geopotential Z at 700 mbar pressure surface', 'U250': 'Zonal wind at 250 mbar pressure surface', 'T700': 'Temperature at 700 mbar pressure surface',  'U200': 'Zonal wind at 200 mbar pressure surface', 'V250': 'Meridional wind at 250 mbar pressure surface', 'T500': 'Temperature at 500 mbar pressure surface', 'Z300': 'Geopotential Z at 300 mbar pressure surface', 'T300': 'Temperature at 300 mbar pressure surface'}
    filepath = '/net/pc200023/nobackup/users/thompson/ETHZ/BOOSTED/'
    year = date[:4]
    mon = date[4:6]
    day = date[6:8]
    if var in ['TAS', 'SLP', 'LP', 'CP', 'HUM', 'Z500']: 
        files = glob.glob(filepath + 'surface_Z500_' + str(year) + '-' + str(mon) + '-' + str(day) + '*')
        cube_list = iris.load(files, dict_var[var])
    else:
        files = glob.glob(filepath + 'TUVZ_levels_' + str(year) + '-' + str(mon) + '-' + str(day) + '*')
        cube_list = iris.load(files, dict_var[var])
    # merge into one cube, with realisation coord
    for n, each in enumerate(cube_list):
        real = files[n].split('ens')[1].split('i')[0][:3]
        new_coord = iris.coords.AuxCoord(int(real), 'realization')
        each.add_aux_coord(new_coord)
    equalise_attributes(cube_list)
    unify_time_units(cube_list)
    cube_list = cube_list.merge_cube()
    # formatting: lats -180 to 180, and added time coords
    cube_list = roll_cube_180(cube_list)
    iris.coord_categorisation.add_year(cube_list, 'time')
    iris.coord_categorisation.add_month(cube_list, 'time')
    iris.coord_categorisation.add_day_of_month(cube_list, 'time')
    return cube_list


def extract_region(cube, region):
    '''
    '''
    latConstraint = iris.Constraint(latitude=lambda cell: region[1] < cell < region[0])
    lonConstraint = iris.Constraint(longitude=lambda cell: region[3] < cell < region[2])
    return cube.extract(latConstraint & lonConstraint)

def euclidean_distance(model, event):
    '''
    Returns list of D
    Inputs required:
      psi = single cube of JJA psi.
      event = cube of single day of event to match.
      BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
    '''
    d = [] # to be list of all euclidean distances
    for each in model:
        D = []
        a, b, c = np.shape(each)
        XA = event.data.reshape(b*c,1)
        #yrs.append(each.coord('year').points[0])
        XB = each.data.reshape(np.shape(each.data)[0], b*c, 1)
        for Xb in XB:
            D.append(np.sqrt(np.sum(np.square(XA - Xb))))
        d.append(D)
    return d

def euclidean_distance_1cube(model, event):
    '''
    Returns list of D
    Inputs required:
      psi = single cube of JJA psi.
      event = cube of single day of event to match.
      BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
    ''' 
    a, b = np.shape(model)
    XA = event.data.reshape(a*b,1)
    XB = model.data.reshape(a*b,1)
    D = np.sqrt(np.sum(np.square(XA - XB)))
    return D


def euclidean_distance_boosted(cube_list, event):
    '''
    Returns list of D
    Inputs required:
      psi = single cube of JJA psi.
      event = cube of single day of event to match.
      BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
    '''
    d = [] # to be list of all euclidean distances
    for each in cube_list:
        D = []
        a, b, c = np.shape(each)
        XA = event.data.reshape(b*c,1)
        #yrs.append(each.coord('year').points[0])
        XB = each.data.reshape(np.shape(each.data)[0], b*c, 1)
        for Xb in XB:
            D.append(np.sqrt(np.sum(np.square(XA - Xb))))
        d.append(D)
    return d


def analog_dates(d, N):
    flat_list = [item for sublist in d for item in sublist]
    d_limit = np.sort(flat_list)[N*5]
    date_list = []; euc_dist = []; real_list = []
    count = 0
    for R in np.arange(len(d)):
        for day in np.arange(len(d[R])):
            if d[R][day] <= d_limit:
                count+=1; print(count); print(d[R][day])
                date_list.append(day)
                euc_dist.append(d[R][day])
                real_list.append(R)
    return date_list, euc_dist, real_list

    
def list_analogs(date_list, real_list, cube_list):
    event_list = iris.cube.CubeList([])
    for day, R in zip(date_list, real_list):
        print (day, R)
        event_list.append(cube_list[R][day, ...])
        print (event_list[-1].coord('realization'))
    return event_list



def plot_analogs(analogs, event_min, event_max):
    '''
    Plots field 
    '''
    fig, axs = plt.subplots(nrows=5, ncols=6, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    lats=analogs[0].coord('latitude').points
    lons=analogs[0].coord('longitude').points
    # subplots
    all_ax = axs.ravel()
    for i, each in enumerate(analogs):
        c = all_ax[i].contourf(lons, lats, each.data, levels=np.linspace(event_min, event_max, 10), cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='both')
        #plot_box(all_ax[i], reg) # plot box of analog region
        all_ax[i].add_feature(cf.BORDERS)
        all_ax[i].add_feature(cf.COASTLINE)
        all_ax[i].set_title(str(analogs[i].coord('realization').points[0])+': '+str(analogs[i].coord('day_of_month').points[0])+analogs[i].coord('month').points[0]+str(analogs[i].coord('year').points[0]))
    plt.tight_layout()   
    return

def roll_cube_180(cube):
    """Takes a cube which goes longitude 0 to 360 back to -180 to 180."""
    lon = cube.coord("longitude")
    new_cube = cube.copy()
    new_cube.data = np.roll(cube.data, len(lon.points) // 2)
    new_cube.coord("longitude").points = lon.points - 180.
    if new_cube.coord("longitude").bounds is not None:
        new_cube.coord("longitude").bounds = lon.bounds - 180.
    return new_cube


def plot_single_event(cube):
    '''
    Plots field 
    '''
    fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    lats=cube.coord('latitude').points
    lons=cube.coord('longitude').points
    event_min = np.min(cube.data)
    event_max = np.max(cube.data)
    c = axs.contourf(lons, lats, cube.data, levels=np.linspace(event_min, event_max, 10), cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='both')
    axs.add_feature(cf.BORDERS)
    axs.add_feature(cf.COASTLINE)
    plt.tight_layout()   
    return


def remove_neighbours(date_list, real_list, euc_dist):
    '''
    Takes date_list and removes any days within 5 days of another event
    Keeps the event with lowest euclidean distance
    '''
    inds = np.argsort(euc_dist)
    new_date_list = np.array(date_list)[inds]
    new_real_list = np.array(real_list)[inds]
    new_euc_dist = np.array(euc_dist)[inds]
    for i in range(len(new_date_list)-1, 0, -1):
        if new_date_list[i]+1 in new_date_list[:i] and new_real_list[np.where(new_date_list==new_date_list[i])[0][0]] == new_real_list[np.where(new_date_list==new_date_list[i]+1)[0][0]]:
            print(1, new_date_list[i], new_real_list[i])
            new_date_list = np.delete(new_date_list, i)
            new_real_list = np.delete(new_real_list, i)
            new_euc_dist = np.delete(new_euc_dist, i)
        elif new_date_list[i]+2 in new_date_list[:i] and new_real_list[np.where(new_date_list==new_date_list[i])[0][0]] == new_real_list[np.where(new_date_list==new_date_list[i]+2)[0][0]]:
            print(2, new_date_list[i], new_real_list[i])
            new_date_list = np.delete(new_date_list, i)
            new_real_list = np.delete(new_real_list, i)
            new_euc_dist = np.delete(new_euc_dist, i)
        elif new_date_list[i]-1 in new_date_list[:i] and new_real_list[np.where(new_date_list==new_date_list[i])[0][0]] == new_real_list[np.where(new_date_list==new_date_list[i]-1)[0][0]]:
            print(-1, new_date_list[i], new_real_list[i])
            new_date_list = np.delete(new_date_list, i)
            new_real_list = np.delete(new_real_list, i)
            new_euc_dist = np.delete(new_euc_dist, i)
        elif new_date_list[i]-2 in new_date_list[:i] and new_real_list[np.where(new_date_list==new_date_list[i])[0][0]] == new_real_list[np.where(new_date_list==new_date_list[i]-2)[0][0]]:
            print(-2, new_date_list[i], new_real_list[i])
            new_date_list = np.delete(new_date_list, i)
            new_real_list = np.delete(new_real_list, i)
            new_euc_dist = np.delete(new_euc_dist, i)
        else:
            pass
    return new_date_list, new_real_list, new_euc_dist



def pull_out_day(var, real, day, mon, yr):
    '''
    var: 'PSL', 'Z500', 'pr_mm',
    '''
    filepath = '/net/pc200023/nobackup/users/thompson/ETHZ'
    if yr > 2014:
        if real == 11 and var == 'tas':
            files = sorted(glob.glob(filepath+'/'+var+'_ssp370_r'+str(real)+'i1p1_2016-2035.nc'))
        else:
            files = sorted(glob.glob(filepath+'/'+var+'_ssp370_r'+str(real)+'i1p1.2015-2035.nc'))
    else:
        files = sorted(glob.glob(filepath+'/'+var+'_ssp370_r'+str(real)+'i1p1.2005-2014.nc'))
    cube = iris.load_cube(files[0])
    # make -180 to 180
    new_cube = roll_cube_180(cube)
    # extract JJA
    #iris.util.promote_aux_coord_to_dim_coord(each, 'time')
    iris.coord_categorisation.add_year(new_cube, 'time')
    iris.coord_categorisation.add_month(new_cube, 'time')
    iris.coord_categorisation.add_day_of_month(new_cube, 'time')
    iris.coord_categorisation.add_season(new_cube, 'time')
    new_cube = new_cube.extract(iris.Constraint(month=mon)&iris.Constraint(day_of_month=day))
    return new_cube.extract(iris.Constraint(year=yr))



def plot_analogs_pr(analogs, event_min, event_max):
    '''
    Plots field 
    '''
    fig, axs = plt.subplots(nrows=5, ncols=6, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,10))
    lats=analogs[0].coord('latitude').points
    lons=analogs[0].coord('longitude').points
    # subplots
    all_ax = axs.ravel()
    for i, each in enumerate(analogs):
        c = all_ax[i].contourf(lons, lats, each.data, levels=np.linspace(event_min, event_max, 10), cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='both')
        #plot_box(all_ax[i], reg) # plot box of analog region
        all_ax[i].set_ylim([40, 60])
        all_ax[i].set_xlim([-10, 20])
        all_ax[i].add_feature(cf.BORDERS)
        all_ax[i].add_feature(cf.COASTLINE)
        all_ax[i].set_title(str(analogs[i].coord('day_of_month').points[0])+analogs[i].coord('month').points[0]+str(analogs[i].coord('year').points[0]))
    plt.tight_layout()   
    return



### Plotting the boosted runs
def plot_prec_z500(startdate, plotdate, region, ens=0):
    '''
    For a selected initialisation date (startdate) and individual date (plotdate)
    Plots precip field with Z500 overlaid
    startdate: str, format YYYYMMDD
    plotdate: str, format YYYYMMDD
    region: region shown in plot, format [N, S, W, E]
    ens: ensemble number, 0-49, default=0
    '''
    ## Get Z500 cube
    Z500_cube = get_Z500_cube(startdate, plotdate, region, ens)
    ## Get TP cube
    TP_cube = get_TP_cube(startdate, plotdate, region, ens)
    # plot data
    plt.ion(); plt.show()
    fig, axs = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,5))
    lats=TP_cube.coord('latitude').points
    lons=TP_cube.coord('longitude').points
    c = axs.contourf(lons, lats, TP_cube.data, levels=np.linspace(0, 50, 10), cmap = plt.cm.get_cmap('Blues'), transform=ccrs.PlateCarree(), extend='both')
    lats=Z500_cube.coord('latitude').points
    lons=Z500_cube.coord('longitude').points
    c = axs.contour(lons, lats, Z500_cube.data, levels=np.linspace(np.min(Z500_cube.data), np.max(Z500_cube.data), 12), cmap = plt.cm.get_cmap('autumn'), transform=ccrs.PlateCarree(), extend='both')
    axs.add_feature(cf.BORDERS)
    axs.add_feature(cf.COASTLINE)
    return


def get_Z500_cube(startdate, plotdate, region, ens=0):
    '''
    '''
    plotYY = plotdate[:4]
    plotMM = plotdate[4:6]
    plotDD = plotdate[6:]
    Z500_cube = ETHZ_boosted_data('Z500', startdate)
    Z500_cube = extract_region(Z500_cube, region)
    Z500_cube = Z500_cube.extract(iris.Constraint(year=int(plotYY), day_of_month=int(plotDD)))
    Z500_cube = Z500_cube[ens,...]
    return Z500_cube


def get_TP_cube(startdate, plotdate, region, ens):
    '''
    '''
    plotYY = plotdate[:4]
    plotMM = plotdate[4:6]
    plotDD = plotdate[6:]
    CP_cube = ETHZ_boosted_data('CP', startdate)
    LP_cube = ETHZ_boosted_data('LP', startdate)
    TP_cube = CP_cube + LP_cube
    TP_cube = extract_region(TP_cube, region)
    TP_cube = TP_cube.extract(iris.Constraint(year=int(plotYY), day_of_month=int(plotDD)))
    TP_cube = TP_cube[ens-1,...]
    TP_cube = TP_cube * 86400 * 1000  # to mm/day
    return TP_cube


def ETHZ_rerun_data(var, ens):
    '''
    Loads cube of specified date and variable
    
    date: YYYYMMDD, string, options 2007-08-07; 2009-08-08; 2016-07-12
    var: string, from in the dictionary 'dict_var'. Key ones: TAS, Z500, CP, LP, 
    '''
    dict_var = {'TAS': 'Reference height temperature', 'SLP': 'Sea level pressure', 'CP': 'Convective precipitation rate (liq + ice)', 'LP': 'Large-scale (stable) precipitation rate (liq + ice)', 'Z500': 'Geopotential Z at 500 mbar pressure surface', 'HUM': 'Reference height humidity', 'Z200': 'Geopotential Z at 200 mbar pressure surface', 'V200': 'Meridional wind at 200 mbar pressure surface', 'Z700': 'Geopotential Z at 700 mbar pressure surface', 'U250': 'Zonal wind at 250 mbar pressure surface', 'T700': 'Temperature at 700 mbar pressure surface',  'U200': 'Zonal wind at 200 mbar pressure surface', 'V250': 'Meridional wind at 250 mbar pressure surface', 'T500': 'Temperature at 500 mbar pressure surface', 'Z300': 'Geopotential Z at 300 mbar pressure surface', 'T300': 'Temperature at 300 mbar pressure surface'}
    filepath = '/net/pc200023/nobackup/users/thompson/ETHZ/WETvDRY/'
    file = glob.glob(filepath + 'BHISTcmip6.0002008.2007-08-15.ens0' + str(ens) + '.cam.h4.2007-08-15-21600.nc')
    cube = iris.load(file, dict_var[var])[0]
    # add realisation coord
    new_coord = iris.coords.AuxCoord(ens, 'realization')
    cube.add_aux_coord(new_coord)
    # formatting: lats -180 to 180, and added time coords
    cube = roll_cube_180(cube)
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')
    iris.coord_categorisation.add_day_of_month(cube, 'time')
    return cube


def get_rerun_TP_cube(region, ens):
    '''
    '''
    CP_cube = ETHZ_rerun_data('CP', ens)
    LP_cube = ETHZ_rerun_data('LP', ens)
    TP_cube = CP_cube + LP_cube
    TP_cube = extract_region(TP_cube, region)
    TP_cube = TP_cube * 86400 * 1000  # to mm/day
    return TP_cube


def get_CP_LP_cube(startdate, plotdate, region, ens=0):
    '''
    '''
    plotYY = plotdate[:4]
    plotMM = plotdate[4:6]
    plotDD = plotdate[6:]
    CP_cube = ETHZ_boosted_data('CP', startdate)
    LP_cube = ETHZ_boosted_data('LP', startdate)
    CP_cube = extract_region(CP_cube, region)
    CP_cube = CP_cube.extract(iris.Constraint(year=int(plotYY), day_of_month=int(plotDD)))
    CP_cube = CP_cube[ens,...]
    CP_cube = CP_cube * 86400 * 1000  # to mm/day
    LP_cube = extract_region(LP_cube, region)
    LP_cube = LP_cube.extract(iris.Constraint(year=int(plotYY), day_of_month=int(plotDD)))
    LP_cube = LP_cube[ens,...]
    LP_cube = LP_cube * 86400 * 1000  # to mm/day
    return CP_cube, LP_cube


def get_slp_cube(startdate, plotdate, region, ens=0):
    '''
    '''
    plotYY = plotdate[:4]
    plotMM = plotdate[4:6]
    plotDD = plotdate[6:]
    slp_cube = ETHZ_boosted_data('SLP', startdate)
    slp_cube = extract_region(slp_cube, region)
    slp_cube = slp_cube.extract(iris.Constraint(year=int(plotYY), day_of_month=int(plotDD)))
    slp_cube = slp_cube[ens,...]
    return slp_cube
