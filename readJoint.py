from preprocess import init_task_joint
from os.path import exists
import os
from shutil import copy
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_seg(xdata, ydata, zdata, figname):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata)
    ax.scatter3D(xdata[0], ydata[0], zdata[0])
    ax.scatter3D(xdata[len(xdata)-1], ydata[len(xdata)-1], zdata[len(xdata)-1])
    ax.text(xdata[0], ydata[0], zdata[0], 'start', color='green')
    ax.text(xdata[len(xdata)-1], ydata[len(xdata)-1], zdata[len(xdata)-1], 'end', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(figname)
    plt.close()

def writecsv(fsb_folder, data_in, emotion, user, gesture, task, sensor):
    path = fsb_folder + 'segments/'
    data = list(data_in)
    for seg_idx in range(len(data)):
        # data_x, data_y, data_z = data[seg]
        filename = path+sensor+'/joint/'+user+'_'+emotion+'_'+gesture+'_'+task+'_'+str(seg_idx)+'.csv'
        figname = path+sensor+'/'+emotion+'/'+user+'/figures/'+str(seg_idx)+'.png'
        fig_path = path+sensor+'/joint/figures/'
        # with open(filename, 'w') as csvfile: 
        #     csvwriter = csv.writer(csvfile)
        #     data[seg] = np.array(data[seg]).T
        #     csvwriter.writerows(data[seg])
        # plot_seg(data[seg_idx].x.values, data[seg_idx].y.values, data[seg_idx].z.values, figname)
        # drawseg(seg_idx, data[seg_idx], current_path, task, user)
        drawpolar(seg_idx, data[seg_idx], fig_path, emotion, user, gesture)
        data[seg_idx].to_csv(filename)

def drawpolar(seg_idx, data, fig_path, emotion, user, gesture):
    fgsz = 150
    color_schemes = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                      'Wistia', 'binary','cool']
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # data = list(data_in)
    # for seg in range(len(data)):
    fgsz = fgsz/100
    pd = fgsz/4
    filename = fig_path+user+'_'+emotion+'_'+gesture+'_'+task+'_'+str(seg_idx)
    fig, ax = plt.subplots(figsize=(fgsz, fgsz),subplot_kw={'projection': 'polar'})
    for joint in range(6):
        theta = data.iloc[:,joint+1].values
        ref = list(range(len(theta)))
        length = len(theta)
        r = np.arange(0, length, 1)
        ax.scatter(theta, r, c=ref, cmap=color_schemes[joint], label='Joint '+str(joint))
    ax.set_rmax(length)
    # ax.legend()
    # ax.set_rticks([0.25*length, 0.5*length, 0.75*length, length])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    # ax.grid(True)

    ax.set_facecolor('xkcd:black') # black background
    fig.patch.set_facecolor('xkcd:black')
    ax.axis('off')
    plt.tight_layout(pad=-pd)
    plt.savefig(filename + ".png")
    plt.close()

human_names = ['time_step', 'x', 'y', 'z', '_']
# franka_joint_names = ['time_step', 
#                 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7',
#                 'vel1', 'vel2', 'vel3', 'vel4', 'vel5', 'vel6', 'vel7',
#                 'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7']
franka_joint_names = ['time_step','elbow_joint','shoulder_lift_joint','shoulder_pan_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
franka_names = ['x', 'y', 'z']
label_names = ['time_step', 'label']
label_names_2 = ['time_step']

directory = '/home/mistlab/Emotion/'
data_folder = 'data/'
dates = ['20221215']
emotions = ["a", "j", "n", "s", "p"]
user1s = ["u0","u2","u3","u4","u5","u7","u8","u9","u10","u11"]
user2s = ["u2","u7","u8","u9"] #u11 only has p
gesture = "lw"
gestures = ["s","stir","star","tri"]
tasks = ["ref","free"]
# gestures2 = ["drink","knock","wave","throw"] # NO REF
# tasks = ["free"]

# save current user
# for task in tasks:
#     for user in users:
#         end_effector = joint_to_axis(fsb_folder, gesture, user, task, franka_joint_names)
#         print(end_effector)

for date_ in dates:
    fsb_folder = directory + data_folder + date_ + '/'
    for emotion in emotions:
        for task in tasks:
            for user in user1s:
                print("Session:", date_, "User:", user, "Gesture:", gesture, "Task:", task, "Emotion:", emotion)
                human_data, franka_data = init_task_joint(fsb_folder, gesture, task, user, emotion, human_names, franka_joint_names, franka_names, label_names)
                # writecsv(fsb_folder, human_data, task, user, "human")
                writecsv(fsb_folder, franka_data, emotion, user, gesture, task, "ur3e")

for date_ in dates:
    fsb_folder = directory + data_folder + date_ + '/'
    for emotion in emotions:
        for task in tasks:
            for gesture in gestures:
                for user in user2s:
                    label = label_names
                    if user == 'u2':
                        label = label_names_2
                    if emotion == 'a':
                        if user == 'u7':
                            if gesture == 's':
                                if task == 'ref':
                                    label = label_names_2
                    if emotion == 'p':
                        if gesture == 'stir':
                            if user == 'u7':
                                continue
                    print("Session:", date_, "User:", user, "Gesture:", gesture, "Task:", task, "Emotion:", emotion)
                    human_data, franka_data = init_task_joint(fsb_folder, gesture, task, user, emotion+'_'+gesture, human_names, franka_joint_names, franka_names, label)
                    # writecsv(fsb_folder, human_data, task, user, "human")
                    writecsv(fsb_folder, franka_data, emotion, user, gesture, task, "ur3e")
print("csv saved")

                    
# plt.show()