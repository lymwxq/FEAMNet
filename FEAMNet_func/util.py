
import numpy as np
import imageio
import sys
import re


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()

        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:

        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data

def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)
        

def load_LFdata(dir_LFimages):    
    traindata_all=np.zeros((len(dir_LFimages), 512, 512, 9, 9, 3),np.uint8)
    traindata_label=np.zeros((len(dir_LFimages), 512, 512),np.float32)
    
    image_id=0
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        for i in range(81):
            try:
                tmp  = np.float32(imageio.imread('hci_dataset/'+dir_LFimage+'/input_Cam0%.2d.png' % i)) # load LF images(9x9) 
            except:
                print('hci_dataset/'+dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
            traindata_all[image_id,:,:,i//9,i-9*(i//9),:]=tmp  
            del tmp
        try:            
            tmp  = np.float32(read_pfm('hci_dataset/'+dir_LFimage+'/gt_disp_lowres.pfm')) # load LF disparity map
        except:
            print('hci_dataset/'+dir_LFimage+'/gt_disp_lowres.pfm..does not exist' % i )            
        traindata_label[image_id,:,:]=tmp  
        del tmp
        image_id=image_id+1
    return traindata_all, traindata_label


def display_current_output(train_output, traindata_label, iter00, directory_save, train_val='train'):

    sz=len(traindata_label)
    train_output=np.squeeze(train_output)
    if(len(traindata_label.shape)>3 and traindata_label.shape[-1]==9): # traindata
        pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
        train_label482=traindata_label[:,15:-15,15:-15,4,4]
    else: # valdata
        pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
        train_label482=traindata_label[:,15:-15,15:-15]
        
    train_output482=train_output[:,15-pad1_half:482+15-pad1_half,15-pad1_half:482+15-pad1_half]
    
    train_diff=np.abs(train_output482-train_label482)
    train_bp=(train_diff>=0.07)
        
    train_output482_all=np.zeros((2*482,sz*482),np.uint8)        
    train_output482_all[0:482,:]=np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)
    train_output482_all[482:2*482,:]=np.uint8(25*np.reshape(np.transpose(train_output482,(1,0,2)),(482,sz*482))+100)
                      
    imageio.imsave(directory_save+'/'+train_val+'_iter%05d.jpg' % (iter00), np.squeeze(train_output482_all))
    
    return train_diff, train_bp


def make_epiinput(image_path,seq1,image_h,image_w,view_n,RGB):
    traindata_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
    i=0
    if(len(image_path)==1):
        image_path=image_path[0]
        
    for seq in seq1:
        tmp  = np.float32(imageio.imread(image_path+'/input_Cam0%.2d.png' % seq)) 
        traindata_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
    return traindata_tmp


def make_input(image_path, image_h, image_w, view_n):
    RGB = [0.299, 0.587, 0.114]  ## RGB to Gray // 0.299 0.587 0.114

    output_list = []
    for i in range(81):
        if(image_path[:12]=='hci_dataset/'):
            A = make_epiinput(image_path, [i], image_h, image_w, [0], RGB)
        # print(A.shape)
        output_list.append(A)

    return output_list