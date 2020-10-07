from __future__ import (absolute_import, division, print_function)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib
import matplotlib.cm as mpl_cm
from matplotlib import colors
import matplotlib.ticker as ticker

plt.close("all")

def find_nearest(array,value):
    """Return array element (and its index) that is closest to the specified value."""
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def find_nearest_negative(array,value):
    """Return array element (and its index) that is negative and closest to the specified value."""
    temp = array-value
    temp[temp>=0] = 100000
    idx = np.absolute(temp).argmin()
    return idx, array[idx]  

### Constants/parameters ################################################################################################################################################################################################
dataset_name_in = ["0625","11","16","25","35_Nz135","444","0625","25","444","r16168SENS_0degcube","r16168SENS_45degcube","r1298ENS_0degcuboid","r121212ENS_new_45degcuboid"]
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0^\circ$",r"B-25-A-$45^\circ$",r"C-33-A-$0^\circ$",r"C-33-A-$45^\circ$"]
lambdap = [0.0625,0.11,0.16,0.25,0.35,0.444,0.0625,0.25,0.444,0.25,0.25,1/3,1/3]
d_arr = [0,1,2,3,4,5,6,7,8,9,10,11,12] # indexes of dataset choose to inspect.
mult_uv = [1.06889352818,1.12941176471,1.19760479042,1.34736842105,1.56774193548,1.84615384615,1.06889352818,1.34736842105,1.84615384615,1/(1-0.25),1/(1-0.25),1/(1-1/3),1/(1-1/3)] # not exactly 1/(1-lambdap) due to grid
mult_others = [1/(1-0.0625),1.125,1/(1-0.16),1/(1-0.25),1.54122621564,1.8,1/(1-0.0625),1/(1-0.25),1.8,1/(1-0.25),1/(1-0.25),1/(1-1/3),1/(1-1/3)] 
eps = 1 - np.array(lambdap)

### Make arrays to store each geometry for variables which are plotted ##################################################################################################################################################

cd_av_x_array = np.zeros(len(d_arr))
cd_av_y_array = np.zeros(len(d_arr))
drag_x = np.zeros((384,len(d_arr)))
drag_y = np.zeros((384,len(d_arr)))
Cd_x = np.zeros((384,len(d_arr)))
Cd_y = np.zeros((384,len(d_arr)))
Cd_av_x = np.zeros(len(d_arr))
Cd_av_y = np.zeros(len(d_arr))
z_final_u = np.zeros((384,len(d_arr)))
wake_production_arr = np.zeros((384,len(d_arr)))

### Bring in data #######################################################################################################################################################################################################
   
individual_plots = "on"
loop = [9,10]#[0,1,2,3,4,5,6,7,8,9,10,11,12]

for d in loop:
    print("\n",dataset_name[d],"\n")

    ### Negin LES #############################################################      

    if d <= 8:
        if d == 0 or d == 3 or d == 5:
            dataset_path = 'C:/Users/nx902220/OneDrive - University of Reading/Desktop/PhD/UrbanCanopies/CFD_Data/Negin_LES_2020_05_04/staggered/With triple correlations/'
            geometry_s_or_a = 'Staggered'
        if d == 1 or d == 2 or d == 4:
            dataset_path = 'C:/Users/nx902220/OneDrive - University of Reading/Desktop/PhD/UrbanCanopies/CFD_Data/Negin_LES_2020_05_04/staggered/'
            geometry_s_or_a = 'Staggered'
        if d > 5 and d <= 8: 
            dataset_path = 'C:/Users/nx902220/OneDrive - University of Reading/Desktop/PhD/UrbanCanopies/CFD_Data/Negin_LES_2020_05_04/aligned/'
            geometry_s_or_a = 'Aligned'

        mat = scipy.io.loadmat('%sENS_Time_Ave_Lp_%s_Lp0.%s.mat' %(dataset_path,geometry_s_or_a,dataset_name_in[d]))
        
        xu_ens = np.array(mat["xu_ens"])[:,0]
        print("xu_ens:", np.shape(xu_ens))
        yv_ens = np.array(mat["yv_ens"])[:,0]
        print("yv_ens:", np.shape(yv_ens))
        zu_3d = np.array(mat["zu_3d"])[:,0]
        print("zu_3d:", np.shape(zu_3d))
        zw_3d = np.array(mat["zw_3d"])[:,0]
        print("zw_3d:", np.shape(zw_3d))
        u_ens = np.array(mat["u_ens"])
        print("u_ens:", np.shape(u_ens))
        v_ens = np.array(mat["v_ens"])
        print("v_ens:", np.shape(v_ens))
        w_ens = np.array(mat["w_ens"])
        print("w_ens:", np.shape(w_ens))
        uw_ens = np.array(mat["uw_ens"])
        print("uw_ens:", np.shape(uw_ens))
        vw_ens = np.array(mat["vw_ens"])
        print("vw_ens:", np.shape(vw_ens))
        uv_ens = np.array(mat["uv_ens"])
        print("uv_ens:", np.shape(uv_ens))
        uu_ens = np.array(mat["uu_ens"])
        print("uu_ens:", np.shape(uu_ens))
        vv_ens = np.array(mat["vv_ens"])
        print("vv_ens:", np.shape(vv_ens))
        ww_ens = np.array(mat["ww_ens"])
        print("ww_ens:", np.shape(ww_ens))
        p_ens = np.array(mat["p_ens"])
        print("p_ens:", np.shape(p_ens))
        e_ens = np.array(mat["e_ens"])
        print("e_ens:", np.shape(e_ens)) 

        # Calculate stresses
        uw_ens = uw_ens - u_ens*w_ens
        vw_ens = vw_ens - v_ens*w_ens
        uv_ens = uv_ens - u_ens*v_ens
        uu_ens = uu_ens - u_ens*u_ens
        vv_ens = vv_ens - v_ens*v_ens
        ww_ens = ww_ens - w_ens*w_ens
        
        # pressure is the modified pressure
        p_ens = p_ens + (2/3)*e_ens

        h = 16
        H = zw_3d[-1] # need to define myself as some datasets don't contain h
            
        # scale grid
        zu_3d = zu_3d / h
        zw_3d = zw_3d / h
        xu_ens = xu_ens / h
        yv_ens = yv_ens / h
        
        # Define u_tau. Don't use np.sqrt(-dpdx*H) since I do not think dpdx=-0.00038281251 for all geometries
        reg_s_idx = find_nearest(zu_3d,2.0)[0]
        reg_e_idx = find_nearest(zu_3d,7.0)[0]
        stress_av = np.mean(uw_ens,axis=tuple(range(0,2)))
        m,b = np.polyfit(zu_3d[reg_s_idx:reg_e_idx+1], stress_av[reg_s_idx:reg_e_idx+1], 1)
        u_tau = np.sqrt(-b)  
        line = m*zu_3d + b
        
        # scale momentum quantities
        u_ens = u_ens / u_tau
        v_ens = v_ens / u_tau
        w_ens = w_ens / u_tau
        uw_ens = uw_ens / u_tau**2
        vw_ens = vw_ens / u_tau**2
        uv_ens = uv_ens / u_tau**2
        uu_ens = uu_ens / u_tau**2
        vv_ens = vv_ens / u_tau**2
        ww_ens = ww_ens / u_tau**2   
        p_ens = p_ens / (u_tau**2/h) * (h/H)
        dpdx = -(u_tau**2/H) / (u_tau**2/h)
        H = H/h
        h = h/h
        
    ### Omduth DNS ############################################################ 
    
    if d >= 9:
        
        dataset_path = 'C:/Users/nx902220/OneDrive - University of Reading/Desktop/PhD/UrbanCanopies/CFD_Data/Momentum/'
        mat = scipy.io.loadmat('%s%s.mat' %(dataset_path,dataset_name_in[d]))
        print(mat.keys())
       
        # reconstruct grid z
        KK = np.array(mat["KK"])
        delt = 1.0/32.0 # DNS grid spacing
        KKz = KK[0,0,:]
        zu_3d = (KKz-0.5)*delt
        zw_3d = np.copy(zu_3d)
        print("zu_3d:", np.shape(zu_3d))
     
        # reconstruct grid y
        JJ = np.array(mat["JJ"])
        delt = 1.0/32.0 # DNS grid spacing
        JJy = JJ[0,:,0]
        yv_ens = (JJy-0.5)*delt
        print("yv_ens:", np.shape(yv_ens))
        
        # reconstruct grid x
        II = np.array(mat["II"])
        delt = 1.0/32.0 # DNS grid spacing
        IIx = II[:,0,0]
        xu_ens = (IIx-0.5)*delt
        print("xu_ens:", np.shape(xu_ens))
        
        u_ens = np.array(mat["U"])
        print("u_ens:", np.shape(u_ens))       
        v_ens = np.array(mat["V"])
        print("v_ens:", np.shape(v_ens))
        w_ens = np.array(mat["W"])
        print("w_ens:", np.shape(w_ens))
        uw_ens = np.array(mat["UW"])
        print("uw_ens:", np.shape(uw_ens))
        vw_ens = np.array(mat["VW"])
        print("vw_ens:", np.shape(vw_ens))
        uv_ens = np.array(mat["UV"])
        print("uv_ens:", np.shape(uv_ens))
        uu_ens = np.array(mat["U2"])
        print("uu_ens:", np.shape(uu_ens)) 
        vv_ens = np.array(mat["V2"])
        print("vv_ens:", np.shape(vv_ens)) 
        ww_ens = np.array(mat["W2"])
        print("ww_ens:", np.shape(ww_ens))
        p_ens = np.array(mat["P"])
        print("p_ens:", np.shape(p_ens))

        h = 1            
        if d == 12: 
            H = 12
        else:
            H = 8
        
        # Define u_tua. It is not 1 for 45 deg.
        if d != 10 and d != 12:
            u_tau = 1
        else:
            reg_s_idx = find_nearest(zu_3d,1.5)[0]
            reg_e_idx = find_nearest(zu_3d,7.0)[0]
            stress_av = np.mean(uw_ens+vw_ens,axis=tuple(range(0,2)))
            m,b = np.polyfit(zu_3d[reg_s_idx:reg_e_idx+1], stress_av[reg_s_idx:reg_e_idx+1], 1)
            u_tau = np.sqrt(-b) 
            
        # scale momentum quantities
        u_ens = u_ens / u_tau
        v_ens = v_ens / u_tau
        w_ens = w_ens / u_tau
        uw_ens = uw_ens / u_tau**2
        vw_ens = vw_ens / u_tau**2
        uv_ens = uv_ens / u_tau**2
        uu_ens = uu_ens / u_tau**2
        vv_ens = vv_ens / u_tau**2
        ww_ens = ww_ens / u_tau**2
        p_ens = p_ens / u_tau**2
        dpdx = -(u_tau**2/H) / (u_tau**2/h)

    ### make volume correction values #############################################
    
    # eps LES
    eps_arr_LES = np.zeros(len(zu_3d)) + 1
    eps_arr_LES[0:32] = eps[d]
    deps_dz_LES = np.zeros(len(zu_3d))
    deps_dz_LES[1:] = (eps_arr_LES[1:]-eps_arr_LES[:-1])/(zu_3d[1:]-zu_3d[:-1])#centred_difference(eps_arr,zu_3d)
    deps_dz_LES[0] = deps_dz_LES[1]
    eps_correction_LES = deps_dz_LES/eps_arr_LES
    #eps DNS
    eps_arr_DNS = np.zeros(len(zu_3d)) + 1
    eps_arr_DNS[0:31] = eps[d]
    deps_dz_DNS = np.zeros(len(zu_3d))
    deps_dz_DNS[1:] = (eps_arr_DNS[1:]-eps_arr_DNS[:-1])/(zu_3d[1:]-zu_3d[:-1])#centred_difference(eps_arr,zu_3d)
    deps_dz_DNS[0] = deps_dz_DNS[1]
    eps_correction_DNS = deps_dz_DNS/eps_arr_DNS

    ### indexes ###################################################################
    
    hidx = find_nearest_negative(zw_3d,1)[0] # index below canopy top
    if d >= 9:
        hidx = hidx - 1

    ### Ensure all of the grid points in the buildings space are zero #############
    
    # find indexes where zero
    u_ens_zero_idxs = u_ens == 0. # u zero idxs
    v_ens_zero_idxs = v_ens == 0. # v zero idxs
    w_ens_zero_idxs = w_ens == 0. # all scalar zero idxs
    
    # set arrays to zero in the building space
    u_ens[u_ens_zero_idxs] = 0.
    v_ens[v_ens_zero_idxs] = 0.
    w_ens[w_ens_zero_idxs] = 0.
    uw_ens[w_ens_zero_idxs] = 0.
    vw_ens[w_ens_zero_idxs] = 0.
    uv_ens[w_ens_zero_idxs] = 0.
    uu_ens[w_ens_zero_idxs] = 0.
    vv_ens[w_ens_zero_idxs] = 0.
    ww_ens[w_ens_zero_idxs] = 0.
    p_ens[w_ens_zero_idxs] = 0.

    ### compute some horizontal spatial averages ##################################
    
    # average in x and y
    Uave = np.mean(u_ens,axis=tuple(range(0,2)))
    Vave = np.mean(v_ens,axis=tuple(range(0,2)))
    Wave = np.mean(w_ens,axis=tuple(range(0,2)))
    uwave = np.mean(uw_ens,axis=tuple(range(0,2)))
    vwave = np.mean(vw_ens,axis=tuple(range(0,2)))
    uvave = np.mean(uv_ens,axis=tuple(range(0,2)))
    uuave = np.mean(uu_ens,axis=tuple(range(0,2)))
    vvave = np.mean(vv_ens,axis=tuple(range(0,2)))
    wwave = np.mean(ww_ens,axis=tuple(range(0,2)))
    Pave = np.mean(p_ens,axis=tuple(range(0,2)))

    # scale due to volume occupied by cube
    for i in np.arange(hidx+2): 
       Uave[i] = mult_uv[d]*Uave[i]
       Vave[i] = mult_uv[d]*Vave[i]
       Wave[i] = mult_others[d]*Wave[i]
       uwave[i] = mult_others[d]*uwave[i]
       vwave[i] = mult_others[d]*vwave[i]
       uvave[i] = mult_others[d]*uvave[i]
       uuave[i] = mult_uv[d]*uuave[i]
       wwave[i] = mult_others[d]*wwave[i]
       vvave[i] = mult_uv[d]*vvave[i]
       Pave[i] = mult_others[d]*Pave[i]

    ### Calculate momentum dispersive stress ##################################

    # make arrays for the dispersive stresses
    Udis=np.zeros(np.shape(u_ens))
    Vdis=np.zeros(np.shape(v_ens))
    Wdis=np.zeros(np.shape(w_ens))
    Pdis=np.zeros(np.shape(p_ens))
    
    # calculate dispersive u, v and w
    Udis = u_ens - Uave
    Vdis = v_ens - Vave
    Wdis = w_ens - Wave
    Pdis = p_ens - Pave
    # Udis and Vdis put onto w grid
    if d <= 8:
        for i in np.arange(len(zu_3d)-1):
            Udis[:,:,i] = (Udis[:,:,i+1] + Udis[:,:,i])/2
            Vdis[:,:,i] = (Vdis[:,:,i+1] + Vdis[:,:,i])/2
            Pdis[:,:,i] = (Pdis[:,:,i+1] + Pdis[:,:,i])/2
        Udis[:,:,-1] = Udis[:,:,-2]
        Vdis[:,:,-1] = Vdis[:,:,-2]
        Pdis[:,:,-1] = Pdis[:,:,-2]
    
    # make mask where u, v, w = 0 i.e. the building space
    u_zero_idxs = u_ens == 0.
    v_zero_idxs = v_ens == 0.
    w_zero_idxs = w_ens == 0.
    p_zero_idxs = p_ens == 0.
    # set to zero in the building space
    Udis[u_zero_idxs] = 0
    Vdis[v_zero_idxs] = 0
    Wdis[w_zero_idxs] = 0
    Pdis[p_zero_idxs] = 0
    
    # calculate dispersive uw and vw
    mult_udiswdis = np.multiply(Udis,Wdis)
    mult_vdiswdis = np.multiply(Vdis,Wdis)
    disuw = np.nanmean(mult_udiswdis,axis=tuple(range(0,2)))
    disvw = np.nanmean(mult_vdiswdis,axis=tuple(range(0,2)))
    # scale due to volume occupied by cube
    for i in np.arange(hidx+2):
       disuw[i] = mult_others[d]*disuw[i]
       disvw[i] = mult_others[d]*disvw[i]
       
    # remove nan at surface
    disuw[0] = 0.
    disvw[0] = 0.
            
    ### plot pressure cross-section ###############################################

    # meshgrid for y-z cross-section
    Yyz=np.zeros([len(yv_ens),len(zw_3d)])
    Z=np.zeros([len(yv_ens),len(zw_3d)])
    for i in np.arange(len(yv_ens)):
        for j in np.arange(len(zw_3d)):
            Yyz[i,j] = yv_ens[i]
            Z[i,j] = zu_3d[j]
    
    # indexes for plotting
    if d == 0:
        # cube 1
        front_idx_1 = 127
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 32
        top_side_1 = bot_side_1 + 31
        # cube 2
        front_idx_2 = -1
        back_idx_2 = front_idx_2 + 34
        bot_side_2 = 96
        top_side_2 = bot_side_2 + 31
    elif d == 1:
        # cube 1
        front_idx_1 = 95
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 16
        top_side_1 = bot_side_1 + 31
        # cube 2
        front_idx_2 = -1
        back_idx_2 = front_idx_2 + 34
        bot_side_2 = 64
        top_side_2 = bot_side_2 + 31
    elif d == 2:
        # cube 1
        front_idx_1 = 79
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 8
        top_side_1 = bot_side_1 + 31
        # cube 2
        front_idx_2 = -1
        back_idx_2 = front_idx_2 + 34
        bot_side_2 = 48
        top_side_2 = bot_side_2 + 31
    elif d == 3:
        # cube 1
        front_idx_1 = 63
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 0
        top_side_1 = bot_side_1 + 31
        # cube 2
        front_idx_2 = -1
        back_idx_2 = front_idx_2 + 34
        bot_side_2 = 32
        top_side_2 = bot_side_2 + 31
    elif d == 4:
        # cube 1
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 22
        top_side_1 = bot_side_1 + 31
    elif d == 5:
        # cube 1
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 16
        top_side_1 = bot_side_1 + 31
    elif d == 6:
        # cube 1
        front_idx_1 = 47
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 48
        top_side_1 = bot_side_1 + 31
    elif d == 7:
        # cube 1
        front_idx_1 = 15
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 16
        top_side_1 = bot_side_1 + 31
    elif d == 8:
        # cube 1
        front_idx_1 = 7
        back_idx_1 = front_idx_1 + 34
        bot_side_1 = 8
        top_side_1 = bot_side_1 + 31
    elif d == 9:
        # cube 1
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 33
        bot_side_1 = 0
        top_side_1 = bot_side_1 + 31
    elif d == 10:
        # cube 1
        # x
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 33
        bot_side_1 = 0
        top_side_1 = bot_side_1 + 31
        # y
        front_idx_1y = 0
        back_idx_1y = front_idx_1 + 33
        bot_side_1y = -1
        top_side_1y = bot_side_1 + 32
    elif d == 11:
        # cube 1
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 33
        bot_side_1 = 0
        top_side_1 = bot_side_1 + 63
    elif d == 12:
        # cube 1
        # x
        front_idx_1 = -1
        back_idx_1 = front_idx_1 + 33
        bot_side_1 = 0
        top_side_1 = bot_side_1 + 63
        # y
        front_idx_1y = 0
        back_idx_1y = front_idx_1 + 33
        bot_side_1y = -1
        top_side_1y = bot_side_1 + 62
        
    ### p cross-section plot ########################################################################################################################################################

    if individual_plots == "on":
            
        ### x-y ###################################################################
        
        # create x-y meshgrids
        X_hor_cross=np.zeros([len(xu_ens),len(yv_ens)])
        Y_hor_cross=np.zeros([len(xu_ens),len(yv_ens)])
        for i in np.arange(len(xu_ens)):
            for j in np.arange(len(yv_ens)):
                X_hor_cross[i,j] = xu_ens[i]
                Y_hor_cross[i,j] = yv_ens[j] 
        
        # count objects for plotting rectangle
        mask = u_ens[:,:,10] == 0.
        array_with_object_numbers = np.zeros(np.shape(mask))
        object_number = 0
        for m in np.arange(len(xu_ens)):
            for n in np.arange(len(yv_ens)):
                if mask[m,n] == False: # non-building values
                    array_with_object_numbers[m,n] = 0.
                else: # building values
                    if m > 0 and array_with_object_numbers[m-1,n] != 0.: # left
                        array_with_object_numbers[m,n] = array_with_object_numbers[m-1,n]
                    elif m < len(xu_ens)-1 and array_with_object_numbers[m+1,n] != 0.: # right
                        array_with_object_numbers[m,n] = array_with_object_numbers[m+1,n]
                    elif n < len(yv_ens)-1 and array_with_object_numbers[m,n+1] != 0.: # above
                        array_with_object_numbers[m,n] = array_with_object_numbers[m,n+1]
                    elif n > 0 and array_with_object_numbers[m,n-1] != 0.: # below
                        array_with_object_numbers[m,n] = array_with_object_numbers[m,n-1]
                    else:
                        object_number += 1
                        array_with_object_numbers[m,n] = object_number
        # number of objects
        num_objects = int(np.max(array_with_object_numbers))
        # get rectangle parameters
        xidxs = np.zeros(num_objects)
        yidxs = np.zeros(num_objects)
        width = np.zeros(num_objects)
        height = np.zeros(num_objects)
        for m in np.arange(1,num_objects+1):
            # coordinate of rectangle bottom left
            idxs = np.where(array_with_object_numbers == m)
            xidxs[m-1] = xu_ens[np.min(idxs[0])]
            yidxs[m-1] = yv_ens[np.min(idxs[1])]
            # width of rectangle
            width[m-1] = xu_ens[np.max(idxs[0])] - xidxs[m-1]
            # height of rectangle
            height[m-1] = yv_ens[np.max(idxs[1])] - yidxs[m-1] 
        
        zidx = 5
        cross = p_ens[:,:,zidx]
        # plot
        fig, ax = plt.subplots(figsize=(10,10))
        cmap = mpl_cm.get_cmap('seismic')
        levels = np.linspace(-np.max(np.abs(cross)),np.max(np.abs(cross)),80)
        CS = ax.contourf(X_hor_cross, Y_hor_cross, cross, cmap = cmap, levels = levels)
        for m in np.arange(0,num_objects):    
            rect = matplotlib.patches.Rectangle((xidxs[m],yidxs[m]),width[m],height[m],linewidth=1,edgecolor='black',facecolor='none')
            #ax.add_patch(rect)
        ticks = np.linspace(levels[0],levels[-1],9)
        cbar = plt.colorbar(CS,label=r"$\overline{p}/u_{\tau}^2$", fraction=0.03, pad=0.04, ticks = ticks)
        cbar.ax.set_yticklabels(["{:3.4f}".format(t) for t in ticks])
        ax.set_aspect('equal')
        ax.set_xlabel("x/h")
        ax.set_ylabel("y/h")
        # fornts and backs 
        plt.axvline(xu_ens[front_idx_1],color="black",linestyle="--")
        plt.axvline(xu_ens[back_idx_1],color="black",linestyle="--")
        if d < 4: # two cubes
            plt.axvline(xu_ens[front_idx_2],color="black",linestyle="--")
            plt.axvline(xu_ens[back_idx_2],color="black",linestyle="--")                 
        # sides
        plt.axhline(yv_ens[bot_side_1],color="black",linestyle="--")
        plt.axhline(yv_ens[top_side_1],color="black",linestyle="--")
        if d < 4: # two cubes
            plt.axhline(yv_ens[bot_side_2],color="black",linestyle="--")
            plt.axhline(yv_ens[top_side_2],color="black",linestyle="--")         
        plt.title(dataset_name[d])
        plt.tight_layout()
       
    ### calculate -1/rho <d\tilde{p}/dx_i>, cd and cd_av ######################

    lambdaf_x = [0.0625,0.11,0.16,0.25,0.35,0.444,0.0625,0.25,0.444,0.25,0.25,1/3,1/3]
    lambdaf_y = [0.0625,0.11,0.16,0.25,0.35,0.444,0.0625,0.25,0.444,0.25,0.25,1/6,1/6]

    dp_x = np.zeros(len(Uave))    
    dp_y = np.zeros(len(Uave))
    form_drag_x = np.zeros(len(Uave))
    form_drag_y = np.zeros(len(Uave))
    cd_x = np.zeros(len(Uave))
    cd_av_x = np.zeros(len(Uave))    
    cd_y = np.zeros(len(Uave))
    cd_av_y = np.zeros(len(Uave))

    for i in np.arange(hidx+2):
        # x
        dp_x[i] = np.nanmean(p_ens[back_idx_1,bot_side_1:top_side_1,i] - p_ens[front_idx_1,bot_side_1:top_side_1,i])
        form_drag_x[i] = -dp_x[i]*lambdaf_x[d]/(h*(1-lambdap[d]))
        cd_x[i] = -dp_x[i]/Uave[i]**2
        if d == 10 or d == 12:
            # y
            dp_y[i] = np.nanmean(p_ens[front_idx_1y:back_idx_1y,top_side_1y,i] - p_ens[front_idx_1y:back_idx_1y,bot_side_1y,i])
            form_drag_y[i] = -dp_y[i]*lambdaf_y[d]/(h*(1-lambdap[d]))
            cd_y[i] = -dp_y[i]/Vave[i]**2

    ### calculate gradients of stresses to plot momentum balance ##############
    
    uwave_dz = np.zeros(len(zw_3d))
    vwave_dz = np.zeros(len(zw_3d))
    disuw_dz = np.zeros(len(zw_3d))
    disvw_dz = np.zeros(len(zw_3d))
    
    # turbulent
    uwave_dz[0:-1] = (uwave[1:]-uwave[0:-1])/(zw_3d[1:]-zw_3d[0:-1])
    uwave_dz[-1] = uwave_dz[-2]
    vwave_dz[0:-1] = (vwave[1:]-vwave[0:-1])/(zw_3d[1:]-zw_3d[0:-1])
    vwave_dz[-1] = vwave_dz[-2]  
    # dispersive
    disuw_dz[0:-1] = (disuw[1:]-disuw[0:-1])/(zw_3d[1:]-zw_3d[0:-1])
    disuw_dz[-1] = disuw_dz[-2]
    disvw_dz[0:-1] = (disvw[1:]-disvw[0:-1])/(zw_3d[1:]-zw_3d[0:-1])
    disvw_dz[-1] = disvw_dz[-2]  
    
    ### plot momentum balance #################################################
    
    if d == 10 or d == 12:
        cp_grad = np.zeros(len(zw_3d)) + dpdx/np.sqrt(2) # constant pressure gradient array
    else:
        cp_grad = np.zeros(len(zw_3d)) + dpdx # constant pressure gradient array
    
    # calculate terms due to change in lambda_p with height
    epsd = 1 - lambdap[d]
    # turbulent
    uw_eps_dz = np.zeros(len(zw_3d))
    uw_eps_dz[hidx+1] = uwave[hidx+1]/((1+epsd)/2) * (1-epsd)/(zw_3d[hidx+2]-zw_3d[hidx+1])
    vw_eps_dz = np.zeros(len(zw_3d))
    vw_eps_dz[hidx+1] = vwave[hidx+1]/((1+epsd)/2) * (1-epsd)/(zw_3d[hidx+2]-zw_3d[hidx+1])   
    # dispersive
    disuw_eps_dz = np.zeros(len(zw_3d))
    disuw_eps_dz[hidx+1] = disuw[hidx+1]/((1+epsd)/2) * (1-epsd)/(zw_3d[hidx+2]-zw_3d[hidx+1])
    disvw_eps_dz = np.zeros(len(zw_3d))
    disvw_eps_dz[hidx+1] = disvw[hidx+1]/((1+epsd)/2) * (1-epsd)/(zw_3d[hidx+2]-zw_3d[hidx+1])  
    
    # remainder of terms
    remainder_x = cp_grad+uwave_dz+disuw_dz+uw_eps_dz+disuw_eps_dz
    remainder_y = cp_grad+vwave_dz+disvw_dz+vw_eps_dz+disvw_eps_dz
    
    if individual_plots == "on":
        # x
        plt.figure()
        plt.plot(-cp_grad,zw_3d,label="cp_grad")
        plt.plot(-uwave_dz,zw_3d,label="uwave_dz")
        plt.plot(-disuw_dz,zw_3d,label="disuw_dz")
        plt.plot(-uw_eps_dz,zw_3d,label="uw_eps_dz")
        plt.plot(-disuw_eps_dz,zw_3d,label="disuw_eps_dz")
        plt.plot(remainder_x,zw_3d,label=r"remainder $\sim$ drag")
        plt.plot(-form_drag_x,zw_3d,label=r"form_drag_x")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-3,3)
        plt.ylim(0,1.5)
        plt.xlabel("momentum equation term, x")
        plt.ylabel("z/h")
        plt.legend(fontsize="10")
        plt.tight_layout()
        # y
        plt.figure()
        plt.plot(-cp_grad,zw_3d,label="cp_grad")
        plt.plot(-vwave_dz,zw_3d,label="vwave_dz")
        plt.plot(-disvw_dz,zw_3d,label="disvw_dz")
        plt.plot(-vw_eps_dz,zw_3d,label="vw_eps_dz")
        plt.plot(-disvw_eps_dz,zw_3d,label="disvw_eps_dz")
        plt.plot(remainder_y,zw_3d,label=r"remainder $\sim$ drag")
        plt.plot(-form_drag_y,zw_3d,label=r"form_drag_y")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-3,3)
        plt.ylim(0,1.5)
        plt.xlabel("momentum equation term, y")
        plt.ylabel("z/h")
        plt.legend(fontsize="10")
        plt.tight_layout()
    
    ### plot integrated momentum balance ######################################
    
    cp_int = np.zeros(len(zw_3d)) # integral of constant pressure gradient array
    
    # calculate terms due to change in lambda_p with height
    uw_eps_int = np.zeros(len(zw_3d))
    vw_eps_int = np.zeros(len(zw_3d))
    disuw_eps_int = np.zeros(len(zw_3d))
    disvw_eps_int = np.zeros(len(zw_3d))
    for i in -np.arange(len(zw_3d))+len(zw_3d)-1:
        cp_int[i-1] = -cp_grad[i-1]*(zw_3d[i-1]-zw_3d[i]) + cp_int[i]
        uw_eps_int[i-1] = -uw_eps_dz[i-1]*(zw_3d[i-1]-zw_3d[i]) + uw_eps_int[i]
        vw_eps_int[i-1] = -vw_eps_dz[i-1]*(zw_3d[i-1]-zw_3d[i]) + vw_eps_int[i]   
        disuw_eps_int[i-1] = -disuw_eps_dz[i-1]*(zw_3d[i-1]-zw_3d[i]) + disuw_eps_int[i]
        disvw_eps_int[i-1] = -disvw_eps_dz[i-1]*(zw_3d[i-1]-zw_3d[i]) + disvw_eps_int[i]      

    # remainder of terms
    remainder_int_x = cp_int-uwave+disuw+uw_eps_int+disuw_eps_int
    remainder_int_y = cp_int-vwave+disvw+vw_eps_int+disvw_eps_int
      
    if individual_plots == "on":
        # x
        plt.figure()
        plt.plot(-cp_int,zw_3d,label="cp_int")
        plt.plot(-uwave,zw_3d,label="uwave")
        plt.plot(disuw,zw_3d,label="disuw")
        plt.plot(-uw_eps_int,zw_3d,label="uw_eps_int")
        plt.plot(-disuw_eps_int,zw_3d,label="disuw_eps_int")
        plt.plot(remainder_int_x,zw_3d,label=r"remainder $\sim$ int drag")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-2.0,2.0)
        plt.ylim(0,1.5)
        plt.xlabel("integrated momentum equation term, x")
        plt.ylabel("z/h")
        plt.legend(fontsize="10")
        plt.tight_layout()
        # y
        plt.figure()
        plt.plot(-cp_int,zw_3d,label="cp_grad")
        plt.plot(-vwave,zw_3d,label="vwave")
        plt.plot(disvw,zw_3d,label="disvw")
        plt.plot(-vw_eps_int,zw_3d,label="vw_eps_int")
        plt.plot(-disvw_eps_int,zw_3d,label="disvw_eps_int")
        plt.plot(remainder_int_y,zw_3d,label=r"remainder $\sim$ int drag")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-2.0,2.0)
        plt.ylim(0,1.5)
        plt.xlabel("integrated momentum equation term, y")
        plt.ylabel("z/h")
        plt.legend(fontsize="10")
        plt.tight_layout()

    ### Depth averaged drag coeffient #########################################

    # x
    av_dp_x = 0
    av_u2_x = 0
    # y
    av_dp_y = 0
    av_u2_y = 0
    # x
    for i in np.arange(hidx+2):
        av_dp_x += dp_x[i]
        av_u2_x += Uave[i]**2
        # y
        if d == 10 or d == 12:
            av_dp_y += dp_y[i]
            av_u2_y += Vave[i]**2
    # x 
    av_dp_x = av_dp_x / i
    av_u2_x = av_u2_x / i
    cd_av_x = -av_dp_x / av_u2_x
    # y 
    av_dp_y = av_dp_y / i
    av_u2_y = av_u2_y / i
    cd_av_y = -av_dp_y / av_u2_y
    
    ### Plot drag coeffients ##################################################
    
    if individual_plots == "on":
        # x
        plt.figure()
        plt.plot(cd_x,zw_3d,marker="x",color="blue")
        plt.axvline(cd_av_x,linestyle="-",color="blue")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-1.0,10.0)
        plt.ylim(0,1.5)
        plt.xlabel("cd_x")
        plt.ylabel("z/h")
        plt.tight_layout()
        # y
        plt.figure()
        plt.plot(cd_y,zw_3d,marker="x",color="blue")
        plt.axvline(cd_av_y,linestyle="-",color="blue")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-1.0,10.0)
        plt.ylim(0,1.5)
        plt.xlabel("cd_y")
        plt.ylabel("z/h")
        plt.tight_layout()

    ### Calculate wake production #############################################

    ### Calculate momentum dispersive stress

    # make arrays for the dispersive stresses
    UWdis=np.zeros(np.shape(uw_ens))
    VWdis=np.zeros(np.shape(vw_ens))
    UVdis=np.zeros(np.shape(uv_ens))
    UUdis=np.zeros(np.shape(uu_ens))
    VVdis=np.zeros(np.shape(vv_ens))
    WWdis=np.zeros(np.shape(ww_ens))
        
    # calculate dispersive
    UWdis = uw_ens - uwave
    VWdis = vw_ens - vwave
    UVdis = uv_ens - uvave
    UUdis = uu_ens - uuave
    VVdis = vv_ens - vvave
    WWdis = ww_ens - wwave
    
    # set to zero in the building space
    UWdis[w_ens_zero_idxs] = 0.
    VWdis[w_ens_zero_idxs] = 0.
    UVdis[w_ens_zero_idxs] = 0.
    UUdis[w_ens_zero_idxs] = 0.
    VVdis[w_ens_zero_idxs] = 0.
    WWdis[w_ens_zero_idxs] = 0.

    ### Calculate vertical gradient of dispersive velocity
    
    dUdis_dz = np.zeros(np.shape(Udis))
    dVdis_dz = np.zeros(np.shape(Vdis))
    dWdis_dz = np.zeros(np.shape(Wdis))
    
    dUdis_dz[:,:,0:-1] = (Udis[:,:,1:]-Udis[:,:,0:-1])/(zu_3d[1:]-zu_3d[0:-1])    
    dVdis_dz[:,:,0:-1] = (Vdis[:,:,1:]-Vdis[:,:,0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    dWdis_dz[:,:,0:-1] = (Wdis[:,:,1:]-Wdis[:,:,0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    
    ### Calculate x gradient of dispersive velocity
    
    dUdis_dx = np.zeros(np.shape(Udis))
    dVdis_dx = np.zeros(np.shape(Vdis))
    dWdis_dx = np.zeros(np.shape(Wdis))

    xu_3d = np.zeros(np.shape(Udis))
    for i in np.arange(len(xu_3d[0,:,0])):
        for j in np.arange(len(xu_3d[0,0,:])):
            xu_3d[:,i,j] = xu_ens

    dUdis_dx[0:-1,:,:] = (Udis[1:,:,:]-Udis[0:-1,:,:])/(xu_3d[1:,:,:]-xu_3d[0:-1,:,:])    
    dVdis_dx[0:-1,:,:] = (Vdis[1:,:,:]-Vdis[0:-1,:,:])/(xu_3d[1:,:,:]-xu_3d[0:-1,:,:])
    dWdis_dx[0:-1,:,:] = (Wdis[1:,:,:]-Wdis[0:-1,:,:])/(xu_3d[1:,:,:]-xu_3d[0:-1,:,:])

    ### Calculate y gradient of dispersive velocity
    
    dUdis_dy = np.zeros(np.shape(Udis))
    dVdis_dy = np.zeros(np.shape(Vdis))
    dWdis_dy = np.zeros(np.shape(Wdis))

    yv_3d = np.zeros(np.shape(Udis))
    for i in np.arange(len(yv_3d[:,0,0])):
        for j in np.arange(len(yv_3d[0,0,:])):
            yv_3d[i,:,j] = yv_ens

    dUdis_dy[:,0:-1,:] = (Udis[:,1:,:]-Udis[:,0:-1,:])/(yv_3d[:,1:,:]-yv_3d[:,0:-1,:])    
    dVdis_dy[:,0:-1,:] = (Vdis[:,1:,:]-Vdis[:,0:-1,:])/(yv_3d[:,1:,:]-yv_3d[:,0:-1,:])
    dWdis_dy[:,0:-1,:] = (Wdis[:,1:,:]-Wdis[:,0:-1,:])/(yv_3d[:,1:,:]-yv_3d[:,0:-1,:])
    
    ### Calculate z gradient of dispersive velocity
    
    dUdis_dz = np.zeros(np.shape(Udis))
    dVdis_dz = np.zeros(np.shape(Vdis))
    dWdis_dz = np.zeros(np.shape(Wdis))

    ZU_3d = np.zeros(np.shape(Udis))
    for i in np.arange(len(ZU_3d[:,0,0])):
        for j in np.arange(len(ZU_3d[0,:,0])):
            ZU_3d[i,j,:] = zu_3d

    dUdis_dz[:,:,0:-1] = (Udis[:,:,1:]-Udis[:,:,0:-1])/(ZU_3d[:,:,1:]-ZU_3d[:,:,0:-1])    
    dVdis_dz[:,:,0:-1] = (Vdis[:,:,1:]-Vdis[:,:,0:-1])/(ZU_3d[:,:,1:]-ZU_3d[:,:,0:-1])
    dWdis_dz[:,:,0:-1] = (Wdis[:,:,1:]-Wdis[:,:,0:-1])/(ZU_3d[:,:,1:]-ZU_3d[:,:,0:-1])

    ### Ensure dispersive velocity gradients are zero in the building space
    
    dUdis_dx[u_ens_zero_idxs] = 0.
    dVdis_dx[v_ens_zero_idxs] = 0.
    dWdis_dx[w_ens_zero_idxs] = 0.
    
    dUdis_dy[u_ens_zero_idxs] = 0.
    dVdis_dy[v_ens_zero_idxs] = 0.
    dWdis_dy[w_ens_zero_idxs] = 0.
    
    dUdis_dz[u_ens_zero_idxs] = 0.
    dVdis_dz[v_ens_zero_idxs] = 0.
    dWdis_dz[w_ens_zero_idxs] = 0.
   
    ### TKE wake production
    
    tke_wake_11 = np.nanmean(UUdis*dUdis_dx,axis=tuple(range(0,2)))
    tke_wake_12 = np.nanmean(UVdis*dUdis_dy,axis=tuple(range(0,2)))
    tke_wake_13 = np.nanmean(UWdis*dUdis_dz,axis=tuple(range(0,2)))
    tke_wake_21 = np.nanmean(UVdis*dVdis_dx,axis=tuple(range(0,2)))
    tke_wake_22 = np.nanmean(VVdis*dVdis_dy,axis=tuple(range(0,2)))
    tke_wake_23 = np.nanmean(VWdis*dVdis_dz,axis=tuple(range(0,2)))
    tke_wake_31 = np.nanmean(UWdis*dWdis_dx,axis=tuple(range(0,2)))
    tke_wake_32 = np.nanmean(VWdis*dWdis_dy,axis=tuple(range(0,2)))
    tke_wake_33 = np.nanmean(WWdis*dWdis_dz,axis=tuple(range(0,2)))
    tke_wake_total = tke_wake_11 + tke_wake_12 + tke_wake_13 + tke_wake_21 + tke_wake_22 + tke_wake_23 + tke_wake_31 + tke_wake_32 + tke_wake_33
    
    # scale due to volume occupied by cube
    for i in np.arange(hidx+2): 
       tke_wake_11[i] = mult_others[d]*tke_wake_11[i]
       tke_wake_12[i] = mult_others[d]*tke_wake_12[i]
       tke_wake_13[i] = mult_others[d]*tke_wake_13[i]
       tke_wake_21[i] = mult_others[d]*tke_wake_21[i]
       tke_wake_22[i] = mult_others[d]*tke_wake_22[i]
       tke_wake_23[i] = mult_others[d]*tke_wake_23[i]
       tke_wake_31[i] = mult_others[d]*tke_wake_31[i]
       tke_wake_32[i] = mult_others[d]*tke_wake_32[i]
       tke_wake_33[i] = mult_others[d]*tke_wake_33[i]
       tke_wake_total[i] = mult_others[d]*tke_wake_total[i]

    ### Calculate <\tilde{u_i}\tilde{u_j}>*d<u_i>/dx_j ########################
        
    # 3D
    mult_udiswdis = np.multiply(Udis,Wdis)
    mult_vdiswdis = np.multiply(Vdis,Wdis)
    # average and scale
    udis_wdis_av = np.nanmean(mult_udiswdis,axis=tuple(range(0,2)))
    vdis_wdis_av = np.nanmean(mult_vdiswdis,axis=tuple(range(0,2)))
    for i in np.arange(hidx+2):
       udis_wdis_av[i] = mult_others[d]*udis_wdis_av[i]
       vdis_wdis_av[i] = mult_others[d]*vdis_wdis_av[i]
    # remove nan at surface
    udis_wdis_av[0] = 0.
    vdis_wdis_av[0] = 0.
    
    # calculate vertical gradient of double averaged velocity
    # x
    dU_dz = np.zeros(np.shape(zu_3d))
    dU_dz[0:-1] = (Uave[1:]-Uave[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    # y
    dV_dz = np.zeros(np.shape(zu_3d))
    dV_dz[0:-1] = (Vave[1:]-Vave[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    
    # calculate final values with the volume fraction correction
    if d <= 8:
        # x
        prod_disp_x = udis_wdis_av*dU_dz + udis_wdis_av*Uave*eps_correction_LES
        # y
        prod_disp_y = vdis_wdis_av*dV_dz + vdis_wdis_av*Vave*eps_correction_LES
    else:
        # x
        prod_disp_x = udis_wdis_av*dU_dz + udis_wdis_av*Uave*eps_correction_DNS
        # y
        prod_disp_y = vdis_wdis_av*dV_dz + vdis_wdis_av*Vave*eps_correction_DNS
    # total
    prod_disp_tot = prod_disp_x + prod_disp_y
    
    ### Calculate viscous term ################################################
    
    # u
    d2Udis_dx2 = np.zeros(np.shape(Udis))
    d2Udis_dy2 = np.zeros(np.shape(Udis))
    d2Udis_dz2 = np.zeros(np.shape(Udis))
    # v 
    d2Vdis_dx2 = np.zeros(np.shape(Vdis))
    d2Vdis_dy2 = np.zeros(np.shape(Vdis))
    d2Vdis_dz2 = np.zeros(np.shape(Vdis))
    
    # u
    d2Udis_dx2[0:-1,:,:] = (dUdis_dx[1:,:,:]-dUdis_dx[0:-1,:,:])/(xu_3d[1:,:,:]-xu_3d[0:-1,:,:])    
    d2Udis_dy2[:,0:-1,:] = (dUdis_dy[:,1:,:]-dUdis_dy[:,0:-1,:])/(yv_3d[:,1:,:]-yv_3d[:,0:-1,:])   
    d2Udis_dz2[:,:,0:-1] = (dUdis_dz[:,:,1:]-dUdis_dz[:,:,0:-1])/(ZU_3d[:,:,1:]-ZU_3d[:,:,0:-1])
    # v
    d2Vdis_dx2[0:-1,:,:] = (dVdis_dx[1:,:,:]-dVdis_dx[0:-1,:,:])/(xu_3d[1:,:,:]-xu_3d[0:-1,:,:])    
    d2Vdis_dy2[:,0:-1,:] = (dVdis_dy[:,1:,:]-dVdis_dy[:,0:-1,:])/(yv_3d[:,1:,:]-yv_3d[:,0:-1,:])   
    d2Vdis_dz2[:,:,0:-1] = (dVdis_dz[:,:,1:]-dVdis_dz[:,:,0:-1])/(ZU_3d[:,:,1:]-ZU_3d[:,:,0:-1])    

    # \tilde{u_i} laplacian \tilde{u_i}
    laplace_3d_u = Udis * (d2Udis_dx2 + d2Udis_dy2 + d2Udis_dz2) 
    laplace_3d_v = Vdis * (d2Vdis_dx2 + d2Vdis_dy2 + d2Vdis_dz2)

    # molecular viscosity 
    if d <= 8:
        scale = u_tau / 16 # u_tau/h 
    else:
        scale = u_tau / 1 # u_tau/h 
    mol_visc = 1.46 * 10**(-5) * scale # I'm not sure what this is for the DNS
    # visocus term
    visc_u = mol_visc * np.nanmean(laplace_3d_u,axis=tuple(range(0,2))) 
    for i in np.arange(hidx+2):
       visc_u[i] = mult_others[d]*visc_u[i]
    visc_v = mol_visc * np.nanmean(laplace_3d_v,axis=tuple(range(0,2))) 
    for i in np.arange(hidx+2):
       visc_v[i] = mult_others[d]*visc_v[i]
    visc_uv = visc_u + visc_v
       
    ### Calculate transport terms #############################################
    
    ### d<\tilde{u_i}\tilde{u_i'w'}>/dz
 
    # 3D
    udis_uwdis = Udis * UWdis
    vdis_vwdis = Vdis * VWdis
    wdis_wwdis = Wdis * WWdis
    # average and scale
    udis_uwdis_av = np.nanmean(udis_uwdis,axis=tuple(range(0,2))) 
    vdis_vwdis_av = np.nanmean(vdis_vwdis,axis=tuple(range(0,2)))
    wdis_wwdis_av = np.nanmean(wdis_wwdis,axis=tuple(range(0,2)))
    for i in np.arange(hidx+2):
       udis_uwdis_av[i] = mult_others[d]*udis_uwdis_av[i]
       vdis_vwdis_av[i] = mult_others[d]*vdis_vwdis_av[i]    
       wdis_wwdis_av[i] = mult_others[d]*wdis_wwdis_av[i]
    # gradient
    d_udis_uwdis_dz = np.zeros(len(zw_3d))
    d_vdis_vwdis_dz = np.zeros(len(zw_3d))
    d_wdis_wwdis_dz = np.zeros(len(zw_3d))    
    d_udis_uwdis_dz[0:-1] = (udis_uwdis_av[1:]-udis_uwdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])   
    d_vdis_vwdis_dz[0:-1] = (vdis_vwdis_av[1:]-vdis_vwdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    d_wdis_wwdis_dz[0:-1] = (wdis_wwdis_av[1:]-wdis_wwdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    # sum plus volume correction terms
    if d <= 8:
        d_uidis_uiwdis_dz = d_udis_uwdis_dz + d_vdis_vwdis_dz + d_wdis_wwdis_dz + (udis_uwdis_av+vdis_vwdis_av+wdis_wwdis_av)*eps_correction_LES
    else:
        d_uidis_uiwdis_dz = d_udis_uwdis_dz + d_vdis_vwdis_dz + d_wdis_wwdis_dz + (udis_uwdis_av+vdis_vwdis_av+wdis_wwdis_av)*eps_correction_DNS

    """
    #Lewis' version.
    # 3D
    wdis_uudis = Wdis * UUdis
    wdis_vvdis = Wdis * VVdis
    wdis_wwdis = Wdis * WWdis
    # average and scale
    wdis_uudis_av = np.nanmean(wdis_uudis,axis=tuple(range(0,2))) 
    wdis_vvdis_av = np.nanmean(wdis_vvdis,axis=tuple(range(0,2)))
    wdis_wwdis_av = np.nanmean(wdis_wwdis,axis=tuple(range(0,2)))
    for i in np.arange(hidx+2):
       wdis_uudis_av[i] = mult_others[d]*wdis_uudis_av[i]
       wdis_vvdis_av[i] = mult_others[d]*wdis_vvdis_av[i]    
       wdis_wwdis_av[i] = mult_others[d]*wdis_wwdis_av[i]
    # gradient
    d_wdis_uudis_dz = np.zeros(len(zw_3d))
    d_wdis_vvdis_dz = np.zeros(len(zw_3d))
    d_wdis_wwdis_dz = np.zeros(len(zw_3d))    
    d_wdis_uudis_dz[0:-1] = (wdis_uudis_av[1:]-wdis_uudis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])   
    d_wdis_vvdis_dz[0:-1] = (wdis_vvdis_av[1:]-wdis_vvdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    d_wdis_wwdis_dz[0:-1] = (wdis_wwdis_av[1:]-wdis_wwdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    # sum
    d_uidis_uiwdis_dz = -(d_wdis_uudis_dz + d_wdis_vvdis_dz + d_wdis_wwdis_dz)/2
    """
    
    ### 0.5*d<\tilde{u_i}\tilde{u_i}\tilde{w}>/dz

    # 3D
    udis_udis_wdis = Udis * Udis * Wdis
    vdis_vdis_wdis = Vdis * Vdis * Wdis
    wdis_wdis_wdis = Wdis * Wdis * Wdis
    # average and scale
    udis_udis_wdis_av = np.nanmean(udis_udis_wdis,axis=tuple(range(0,2))) 
    vdis_vdis_wdis_av = np.nanmean(vdis_vdis_wdis,axis=tuple(range(0,2)))
    wdis_wdis_wdis_av = np.nanmean(wdis_wdis_wdis,axis=tuple(range(0,2)))
    for i in np.arange(hidx+2):
       udis_udis_wdis_av[i] = mult_others[d]*udis_udis_wdis_av[i]
       vdis_vdis_wdis_av[i] = mult_others[d]*vdis_vdis_wdis_av[i]    
       wdis_wdis_wdis_av[i] = mult_others[d]*wdis_wdis_wdis_av[i]
    # gradient
    d_udis_udis_wdis_dz = np.zeros(len(zw_3d))
    d_vdis_vdis_wdis_dz = np.zeros(len(zw_3d))
    d_wdis_wdis_wdis_dz = np.zeros(len(zw_3d))    
    d_udis_udis_wdis_dz[0:-1] = (udis_udis_wdis_av[1:]-udis_udis_wdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])   
    d_vdis_vdis_wdis_dz[0:-1] = (vdis_vdis_wdis_av[1:]-vdis_vdis_wdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    d_wdis_wdis_wdis_dz[0:-1] = (wdis_wdis_wdis_av[1:]-wdis_wdis_wdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1])
    # sum plus volume correction terms
    if d <= 8:
        d_uidis_uidis_wdis_dz = 0.5*(d_udis_udis_wdis_dz + d_vdis_vdis_wdis_dz + d_wdis_wdis_wdis_dz) + 0.5*(udis_udis_wdis_av+vdis_vdis_wdis_av+wdis_wdis_wdis_av)*eps_correction_LES
    else:
        d_uidis_uidis_wdis_dz = 0.5*(d_udis_udis_wdis_dz + d_vdis_vdis_wdis_dz + d_wdis_wdis_wdis_dz) + 0.5*(udis_udis_wdis_av+vdis_vdis_wdis_av+wdis_wdis_wdis_av)*eps_correction_DNS   
     
    ### (1/rho)*d<\tilde{p}\tilde{w}>/dz
    
    # 3D
    pdis_wdis = Pdis * Wdis
    # average and scale
    pdis_wdis_av = np.nanmean(pdis_wdis,axis=tuple(range(0,2))) 
    for i in np.arange(hidx+2):
       pdis_wdis_av[i] = mult_others[d]*pdis_wdis_av[i]
    # gradient
    d_pdis_wdis_dz = np.zeros(len(zw_3d))  
    if d <= 8:
        d_pdis_wdis_dz[0:-1] = (pdis_wdis_av[1:]-pdis_wdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1]) + pdis_wdis_av[0:-1]*eps_correction_LES[0:-1]   
    else:
        d_pdis_wdis_dz[0:-1] = (pdis_wdis_av[1:]-pdis_wdis_av[0:-1])/(zu_3d[1:]-zu_3d[0:-1]) + pdis_wdis_av[0:-1]*eps_correction_DNS[0:-1]

    ### R&S1982 Eq.16 DKE budget (rearranged so that wake production is the only term on the RHS) ###

    # for cd constant drag mutiply velocity
    cd_av_x_arr = np.zeros(len(zw_3d)) 
    cd_av_x_arr[:32] = np.ones(len(zw_3d))[:32]*cd_av_x
    cd_av_y_arr = np.zeros(len(zw_3d)) 
    cd_av_y_arr[:32] = np.ones(len(zw_3d))[:32]*cd_av_y
    
    # form drag multiply velocity
    wake_prod_scheme1 = Uave*form_drag_x + Vave*form_drag_y 
    
    if individual_plots == "on":
        plt.figure(figsize=(6,8))
        matplotlib.rcParams.update({'font.size': 26})
        plt.plot(tke_wake_total,zw_3d,label=r"$\left< \widetilde{u'_iu'}_j \frac{\partial \tilde{u}_i}{\partial x_j} \right>$",color="blue")
        plt.plot(-prod_disp_tot,zw_3d,label=r"$-\left< \tilde{u}_i \tilde{u}_j \right> \partial <\overline{u_i}>/\partial x_j$",color="blue",linestyle="dotted")
        plt.plot(wake_prod_scheme1,zw_3d,label=r"$(1/\rho)<\overline{u_i}>\left< \partial \tilde{p}/\partial x_i \right>$",color="blue",linestyle="dashed",marker="^",markevery=7,markersize=8)
        if d == 10 or d==12:
            plt.plot(cd_av_x_arr*Uave**3*lambdaf_x[d]/(h*(1-lambdap[d]))+cd_av_y_arr*Vave**3*lambdaf_x[d]/(h*(1-lambdap[d])),zw_3d,label=r"$(1/\rho)<\overline{u_i}>\left< \partial \tilde{p}/\partial x_i \right>$",color="blue",linestyle="dashed",marker="v",markevery=7,markersize=8)
        else:
            plt.plot(cd_av_x_arr*Uave**3*lambdaf_x[d]/(h*(1-lambdap[d])),zw_3d,label=r"$(1/\rho)<\overline{u_i}>\left< \partial \tilde{p}/\partial x_i \right>$",color="blue",linestyle="dashed",marker="v",markevery=7,markersize=8)
        plt.plot(-d_uidis_uiwdis_dz,zw_3d,label=r"$-\partial <\tilde{u}_i\widetilde{u'_iu'}_j>/\partial x_j$",color="red")
        plt.plot(-d_uidis_uidis_wdis_dz,zw_3d,label=r"$-\partial (<\tilde{u}_i\tilde{u}_i\tilde{u}_j>/2)/\partial x_j$",color="red",linestyle="dashed")
        plt.plot(-d_pdis_wdis_dz,zw_3d,label=r"$-\partial (<\tilde{p}\tilde{u}_j>/\rho)/\partial x_j$",color="red",linestyle="dotted")
        #plt.plot(visc_uv,zw_3d,label=r"$-\nu\left< \tilde{u}_i \nabla^2 \tilde{u}_i \right>$")
        #plt.plot(-wake_prod_scheme1 + prod_disp_tot - visc_uv + d_uidis_uiwdis_dz + d_uidis_uidis_wdis_dz + d_pdis_wdis_dz,zw_3d,label="wake prod = sum",color="orange")
        plt.axvline(0,linestyle="dotted",color="black")
        plt.xlim(-7.8,11.5)
        plt.xticks([-5,0,5,10])
        plt.ylim(0,1.2)
        plt.xlabel("DKE budget terms")
        plt.ylabel("z/h")
        #plt.legend(fontsize=15.)
        plt.tight_layout()
        if d == 9:
            plt.savefig("DKE_budget_DNS_cube_0deg.png")
        if d == 10:
            plt.savefig("DKE_budget_DNS_cube_45deg.png")        
            
    ### Save to final arrays ##################################################

    cd_av_x_array[d] = cd_av_x   
    cd_av_y_array[d] = cd_av_y
    drag_x[0:len(form_drag_x),d] = form_drag_x
    drag_y[0:len(form_drag_y),d] = form_drag_y
    Cd_x[0:len(form_drag_x),d] = cd_x
    Cd_y[0:len(form_drag_y),d] = cd_y
    Cd_av_x[d] = cd_av_x
    Cd_av_y[d] = cd_av_y
    z_final_u[0:len(zu_3d),d] = zu_3d
    z_final_u[len(zu_3d):,d] = np.max(zu_3d)  
    wake_production_arr[0:len(tke_wake_total),d] = tke_wake_total
    
# LES drag
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0^\circ$",r"B-25-A-$45^\circ$",r"C-33-A-$0^\circ$",r"C-33-A-$45^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","solid","dashed","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [0,1,2,3,4,5,6,7,8]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})
    
for i in loop:
    plt.plot(drag_x[:,i],z_final_u[:,i],marker=marker[i],markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=dataset_name[i])
plt.xlabel(r"$\frac{1}{\rho} \left< \frac{\partial \tilde{p}}{\partial x} \right> \times \frac{h}{u_\tau^2}$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(0,5)
plt.ylim(0.0,1.05)
plt.legend(fontsize=16)
plt.tight_layout()

# DNS drag
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0\!^\circ$",r"B-25-A-$45\!^\circ$",r"C-33-A-$0\!^\circ$",r"C-33-A-$45\!^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","dashed","solid","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [9,10,11,12]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})

for i in loop:
    if i == 9 or i == 11:
        plt.plot(drag_x[:,i],z_final_u[:,i],marker="",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=1$" %dataset_name[i])
    else:
        plt.plot(drag_x[:,i],z_final_u[:,i],marker="v",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=1$" %dataset_name[i])
        plt.plot(drag_y[:,i],z_final_u[:,i],marker="^",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=2$" %dataset_name[i])
plt.xlabel(r"$\frac{1}{\rho} \left< \frac{\partial \tilde{p}}{\partial x_i} \right> \times \frac{h}{u_\tau^2}$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(0,5)
plt.ylim(0.0,1.05)
plt.legend(fontsize=16)
plt.tight_layout()

# LES sectional drag
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0^\circ$",r"B-25-A-$45^\circ$",r"C-33-A-$0^\circ$",r"C-33-A-$45^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","solid","dashed","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [0,1,2,3,4,5,6,7,8]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})
    
for i in loop:
    plt.plot(Cd_x[:,i],z_final_u[:,i],marker=marker[i],markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=dataset_name[i])
plt.xlabel(r"$C_d(z)$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(0,5)
plt.ylim(0.0,1.05)
plt.legend(fontsize=16)
plt.tight_layout()

### DNS sectional drag
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0\!^\circ$",r"B-25-A-$45\!^\circ$",r"C-33-A-$0\!^\circ$",r"C-33-A-$45\!^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","dashed","solid","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [9,10,11,12]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})

for i in loop:
    if i == 9 or i == 11:
        plt.plot(Cd_x[:,i],z_final_u[:,i],marker="",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=1$" %dataset_name[i])
    else:
        plt.plot(Cd_x[:,i],z_final_u[:,i],marker="v",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=1$" %dataset_name[i])
        plt.plot(Cd_y[:,i],z_final_u[:,i],marker="^",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, $i=2$" %dataset_name[i])
plt.xlabel(r"$C_d(z)$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(0,5)
plt.ylim(0.0,1.05)
plt.legend(fontsize=16)
plt.tight_layout()
   
### paper drag coefficient plot ###############################################

### DNS sectional drag
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0\!^\circ$",r"B-25-A-$45\!^\circ$",r"C-33-A-$0\!^\circ$",r"C-33-A-$45\!^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","dashed","solid","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [9,10]

plt.figure(figsize=(6,8))
matplotlib.rcParams.update({'font.size': 26})

for i in loop:
    if i == 9:
        plt.plot(Cd_x[:33,i],z_final_u[:33,i],marker="",markersize=markersize[i],markevery=markevery[i],color="black",linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s" %dataset_name[i])
        plt.axvline(Cd_av_x[i],ymin=0,ymax=1/1.2,color="black",linewidth=linewidth[i],linestyle=linestyle[i])
    else:
        plt.plot((Cd_x[:33,i]+Cd_y[:33,i])/2,z_final_u[:33,i],marker="",markersize=markersize[i],markevery=markevery[i],color="black",linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s" %dataset_name[i])
        plt.axvline(Cd_av_x[i],ymin=0,ymax=1/1.2,color="black",linewidth=linewidth[i],linestyle=linestyle[i])
plt.xlabel(r"$C_d(z)$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(0,7)
plt.ylim(0.0,1.2)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("Cd_0_45.png")

###############################################################################
 
### drag coefficients

dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0^\circ$",r"B-25-A-$45^\circ$",r"C-33-A-$0^\circ$",r"C-33-A-$45^\circ$"]
marker=["x","x","x","x","x","x","*","*","*","o","*","o","*"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["","","","","","","","","","","","",""]
markersize=[10,10,10,10,10,10,10,10,10,10,10,10,10]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

matplotlib.rcParams.update({'font.size': 22})

loop = [0,1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(10,7))
for i in loop:  
    if i == 12:
        #x
        plt.plot(lambdap[i],cd_av_x_array[i],marker="v",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, x" %dataset_name[i])
        #y
        plt.plot(lambdap[i],cd_av_y_array[i],marker="^",markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s, y" %dataset_name[i])        
    else:
        plt.plot(lambdap[i],cd_av_x_array[i],marker=marker[i],markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=r"%s" %dataset_name[i])
plt.xlabel(r"$\lambda_p$")
plt.ylabel(r"$C_d$")
plt.legend(fontsize=15,ncol=2)
plt.tight_layout()

# LES wake
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0^\circ$",r"B-25-A-$45^\circ$",r"C-33-A-$0^\circ$",r"C-33-A-$45^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","solid","dashed","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [0,1,2,3,4,5,6,7,8]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})
    
for i in loop:
    plt.plot(wake_production_arr[:,i],z_final_u[:,i],marker=marker[i],markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=dataset_name[i])
plt.xlabel(r"$\left< \widetilde{u'_iu'}_j \frac{\partial \tilde{u}_i}{\partial x_j} \right> \times \frac{h}{u_\tau^3}$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(-10,10)
plt.ylim(0.0,1.2)
plt.legend(fontsize=16)
plt.tight_layout()

# DNS wake
    
dataset_name = [r"N-0625-S",r"N-11-S",r"N-16-S",r"N-25-S",r"N-35-S",r"N-44-S",r"N-0625-A",r"N-25-A",r"N-44-A",r"B-25-A-$0\!^\circ$",r"B-25-A-$45\!^\circ$",r"C-33-A-$0\!^\circ$",r"C-33-A-$45\!^\circ$"]
marker=["","","","","","","","","","","D","","D"]
color=["darkgoldenrod","darkviolet","forestgreen","blue","maroon","black","darkgoldenrod","blue","black","blue","blue","darkorange","darkorange"]
linestyle=["solid","solid","solid","solid","solid","solid","dashed","dashed","dashed","solid","dashed","solid","dashed"]
markersize=[7,7,7,7,7,7,7,7,7,7,7,7,7]
markevery=[7,7,7,7,7,7,7,7,7,7,7,7,7]
linewidth=[2,2,2,2,2,2,2,2,2,2,2,2,2]

loop = [9,10,11,12]

plt.figure(figsize=(8,8))
matplotlib.rcParams.update({'font.size': 26})

for i in loop:
    plt.plot(wake_production_arr[:,i],z_final_u[:,i],marker=marker[i],markersize=markersize[i],markevery=markevery[i],color=color[i],linewidth=linewidth[i],linestyle=linestyle[i],label=dataset_name[i])
plt.xlabel(r"$\left< \widetilde{u'_iu'}_j \frac{\partial \tilde{u}_i}{\partial x_j} \right> \times \frac{h}{u_\tau^3}$")
plt.ylabel("z/h")
#plt.axhline(1,linestyle="dotted",color="black")
#plt.axvline(0,linestyle="dotted",color="black")
plt.xlim(-10,10)
plt.ylim(0,1.2)
plt.legend(fontsize=16)
plt.tight_layout()   