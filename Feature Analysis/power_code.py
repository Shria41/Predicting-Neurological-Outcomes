import os
import json
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io as sio

sf=100
win = 40 * sf

directory=r"D:\Capstone\i-care-international-cardiac-arrest-research-consortium-database-1.0\training"
cpc4=[340, 357, 369, 403, 441, 493, 551, 579, 597]
cpc2=[6, 10, 16, 17, 30, 39, 52, 60, 84, 93, 99, 102, 123, 135, 148, 150, 160, 191, 199, 212, 220, 228, 234, 257, 280, 302, 303, 313, 314, 320, 344, 362, 363, 404, 435, 448, 468, 511, 524, 545, 571, 582, 598, 600]
cpc1=[0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 18, 23, 26, 27, 28, 31, 33, 38, 41, 43, 49, 58, 62, 66, 68, 74, 79, 81, 82, 83, 85, 91, 95, 101, 103, 106, 107, 111, 116, 118, 121, 122, 127, 128, 133, 136, 137, 141, 144, 151, 152, 155, 157, 158, 161, 162, 164, 180, 194, 195, 197, 200, 202, 205, 206, 208, 216, 221, 222, 223, 227, 229, 231, 232, 235, 238, 240, 248, 253, 254, 255, 261, 264, 266, 275, 276, 279, 281, 282, 284, 286, 295, 304, 305, 307, 309, 310, 315, 317, 322, 327, 331, 333, 336, 339, 345, 346, 351, 358, 360, 361, 364, 367, 371, 372, 374, 375, 383, 384, 395, 396, 399, 413, 414, 416, 421, 422, 425, 426, 428, 431, 436, 439, 442, 444, 446, 449, 450, 451, 453, 454, 456, 460, 461, 463, 466, 469, 478, 482, 486, 488, 491, 501, 508, 509, 515, 521, 526, 528, 529, 535, 540, 541, 543, 544, 546, 555, 560, 568, 575, 581, 586, 587, 590, 604, 605]
cpc3=[21, 47, 76, 179, 181, 183, 203, 215, 236, 262, 267, 311, 341, 373, 391, 434, 443, 527, 547, 561]
cpc5=[19, 20, 22, 24, 25, 29, 32, 34, 35, 36, 37, 40, 42, 44, 45, 46, 48, 50, 51, 53, 54, 55, 56, 57, 59, 61, 63, 64, 65, 67, 69, 70, 71, 72, 73, 75, 77, 78, 80, 86, 87, 88, 89, 90, 92, 94, 96, 97, 98, 100, 104, 105, 108, 109, 110, 112, 113, 114, 115, 117, 119, 120, 124, 125, 126, 129, 130, 131, 132, 134, 138, 139, 140, 142, 143, 145, 146, 147, 149, 153, 154, 156, 159, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 182, 184, 185, 186, 187, 188, 189, 190, 192, 193, 196, 198, 201, 204, 207, 209, 210, 211, 213, 214, 217, 218, 219, 224, 225, 226, 230, 233, 237, 239, 241, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 256, 258, 259, 260, 263, 265, 268, 269, 270, 271, 272, 273, 274, 277, 278, 283, 285, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 306, 308, 312, 316, 318, 319, 321, 323, 324, 325, 326, 328, 329, 330, 332, 334, 335, 337, 338, 342, 343, 347, 348, 349, 350, 352, 353, 354, 355, 356, 359, 365, 366, 368, 370, 376, 377, 378, 379, 380,381, 382, 385, 386, 387, 388, 389, 390, 392, 393, 394, 397, 398, 400, 401, 402, 405, 406, 407, 408, 409, 410, 411, 412, 415, 417, 418, 419, 420, 423, 424, 427, 429, 430, 432, 433, 437, 438, 440, 445, 447, 452, 455, 457, 458, 459, 462, 464, 465, 467, 470, 471, 472, 473, 474, 475, 476, 477, 479, 480, 481, 483, 484, 485, 487, 489, 490, 492, 494, 495, 496, 497, 498, 499, 500, 502, 503, 504, 505, 506, 507, 510, 512, 513, 514, 516, 517, 518, 519, 520, 522, 523, 525, 530, 531, 532, 533, 534, 536, 537, 538, 539, 542, 548, 549, 550, 552, 553, 554, 556, 557, 558, 559, 562, 563, 564, 565, 566, 567, 569, 570, 572, 573, 574, 576, 577, 578, 580, 583, 584, 585, 588, 589, 591, 592, 593, 594, 595, 596, 599, 601, 602, 603]

dict_per_cpc={"alpha":{1:{},2:{},3:{},4:{},5:{}},
              "beta":{1:{},2:{},3:{},4:{},5:{}},
              "delta":{1:{},2:{},3:{},4:{},5:{}},
              "theta":{1:{},2:{},3:{},4:{},5:{}}}

def band(frequency_band,low,high,freqs,psd,cpc,i1):
    

    #delta
    #low, high = 0.5, 4
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs < high)
    from scipy.integrate import simps

    # Frequency resolution  ``
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

    # Compute the absolute power by approximating the area under the curve
    delta_power = simps(psd[idx_delta], dx=freq_res)        
    total_power = simps(psd, dx=freq_res)
    #print(i1)
    if delta_power:
        delta_rel_power = delta_power / total_power
        if(i1+1 not in list(dict_per_cpc[frequency_band][cpc].keys())):
            dict_per_cpc[frequency_band][cpc][i1+1]=[total_power,1,delta_rel_power,1,delta_power,1]
        else:
            dict_per_cpc[frequency_band][cpc][i1+1][0]+=total_power
            dict_per_cpc[frequency_band][cpc][i1+1][1]+=1
            dict_per_cpc[frequency_band][cpc][i1+1][2]+=delta_rel_power
            dict_per_cpc[frequency_band][cpc][i1+1][3]+=1
            dict_per_cpc[frequency_band][cpc][i1+1][4]+=delta_power
            dict_per_cpc[frequency_band][cpc][i1+1][5]+=1
    else:
        if(i1+1 not in list(dict_per_cpc[frequency_band][cpc].keys())):
            dict_per_cpc[frequency_band][cpc][i1+1]=[total_power,1,0,0,delta_power,1]
        else:
            dict_per_cpc[frequency_band][cpc][i1+1][0]+=total_power
            dict_per_cpc[frequency_band][cpc][i1+1][1]+=1
            #dict_per_cpc[frequency_band][cpc][i1+1][2]+=delta_rel_power
            #dict_per_cpc[frequency_band][cpc][i1+1][3]+=1
            dict_per_cpc[frequency_band][cpc][i1+1][4]+=delta_power
            dict_per_cpc[frequency_band][cpc][i1+1][5]+=1
        print('Exception')

def func(data,cpc):
    for i1 in range(18):

        data1=data[i1]
        freqs, psd = signal.welch(data1, sf, nperseg=win)
        band("delta",0.5,4.0,freqs, psd,cpc,i1)
        band("theta",4.0,8.0,freqs, psd,cpc,i1)
        band("alpha",8.0,13.0,freqs, psd,cpc,i1)
        band("beta",13.0,30.0,freqs, psd,cpc,i1)
        
def find_cpc(subdir):
    f = os.path.join(directory, subdir)
    b=r'\\'
    addr=f+b+subdir+'.txt'
    #print(addr)
    f1=open(addr,"r")
    return int(f1.read().split('\n')[8].split(':')[1].strip())

i=0
for subdir in os.listdir(directory): 
    if(subdir.startswith('ICARE')):
        #c=find_cpc(subdir)
        if(i in cpc1):
            f = os.path.join(directory, subdir) #subdir=patient
            mat_files = [file for file in os.listdir(f) if file.endswith('.mat')]
            for j in mat_files:   
                mat_data = sio.loadmat(os.path.join(f,j))
                data1 = mat_data['val']       
                func(data1,1)
        elif(i in cpc2):
            f = os.path.join(directory, subdir) #subdir=patient
            mat_files = [file for file in os.listdir(f) if file.endswith('.mat')]
            for j in mat_files:   
                mat_data = sio.loadmat(os.path.join(f,j))
                data1 = mat_data['val']       
                func(data1,2)
        elif(i in cpc3):
            f = os.path.join(directory, subdir) #subdir=patient
            mat_files = [file for file in os.listdir(f) if file.endswith('.mat')]
            for j in mat_files:   
                mat_data = sio.loadmat(os.path.join(f,j))
                data1 = mat_data['val']       
                func(data1,3)
        elif(i in cpc4):
            f = os.path.join(directory, subdir) #subdir=patient
            mat_files = [file for file in os.listdir(f) if file.endswith('.mat')]
            for j in mat_files:   
                mat_data = sio.loadmat(os.path.join(f,j))
                data1 = mat_data['val']       
                func(data1,4)
        elif(i in cpc5):
            f = os.path.join(directory, subdir) #subdir=patient
            mat_files = [file for file in os.listdir(f) if file.endswith('.mat')]
            for j in mat_files:   
                mat_data = sio.loadmat(os.path.join(f,j))
                data1 = mat_data['val']       
                func(data1,5)
        else:
            print('error for ',i)
            break
        i+=1
        

f=open('dict_bands_win1.txt','w')
f.write(json.dumps(dict_per_cpc))
f.close()