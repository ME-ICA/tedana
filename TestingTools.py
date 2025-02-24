import logging
import typing
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os  
from fnmatch import fnmatch
import subprocess
import sys
import nilearn.image as nli

import subprocess
import nibabel as nib
from nilearn._utils import check_niimg
from nilearn.image import new_img_like


import pandas as pd
import contextlib
import shutil

import tedana.workflows.tedana as tedana_update



tedanaVersion="24.0.2*"



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
def import_h5py():
    modulename = 'h5py'
    if modulename not in sys.modules:
        try:
            import h5py
        except:
            install("h5py")
            import h5py

def niiTohdf5(path):
    import_h5py()
    file_name = os.path.basename(path)
    savepath = os.path.join(os.path.dirname(path),os.path.splitext(file_name)[0] +".h5")
    data=nli.load_img(path).get_fdata()
    f = h5py.File(savepath, "w")
    dset = f.create_dataset("data", np.shape(data))
    data=data.astype('float32') 
    return data

def loadNsaveTohdf5(dataDir,savepath,echosToDownload=None,filePattern=None,fileSortType=1,loadOnly=1):
    echoes = []
    echoPath = dataDir

    tmp = os.listdir(echoPath)
    if filePattern:
        files=[]
        for filename in tmp:
            if fnmatch(filename,filePattern):
                files.append(filename)
    else:
        files=tmp
    if fileSortType==1:
        files.sort(key=fileSortHelper)
    else:
        files.sort(key=fileSortHelperType2)
        
    Ne=len(files)
    Nx,Ny,Nz,Nt = np.shape(nli.load_img(os.path.join(echoPath,files[0])).get_fdata())
    helper=np.zeros((Nx,Ny,Nz,Nt,Ne), dtype=np.float32)
    count=0
    index=0
    for file in files:
        if file[-4:]==".nii" or file[-7:]==".nii.gz":
            if echosToDownload is None or index in echosToDownload:
                helper[:,:,:,:,count]=nli.load_img(os.path.join(echoPath,file)).get_fdata()
                count += 1
            if count%10==0 and count:
                if echosToDownload is None:
                    print("Downloaded {0}/{1}".format(count,len(files)))
                else:
                    print("Downloaded {0}/{1}".format(count,len(echosToDownload)))
        index += 1

    helper=helper.astype('float32') 
    if loadOnly==0:
        import_h5py()
        try:
            f = h5py.File(savepath, "w")
            dset = f.create_dataset("data", np.shape(data))
        except:
            print("Data will take up too much memory. Data not saved to file, but data is returned")
    return helper

def strOfFilePaths(dataDir,echosToDownload=None,filePattern=None,fileSortType=1,isOutStr=0):
    echoes = []
    echoPath = dataDir

    tmp = os.listdir(echoPath)
    if filePattern:
        files=[]
        for filename in tmp:
            if fnmatch(filename,filePattern):
                files.append(filename)
    else:
        files=tmp
    if fileSortType==1:
        files.sort(key=fileSortHelper)
    else:
        files.sort(key=fileSortHelperType2)

    if echosToDownload is not None:
        filesOut=[]
        for index in echosToDownload:
            filesOut.append(files[index-1])
    else:
        filesOut=files
    for i in range(len(filesOut)):
        filesOut[i]=os.path.join(dataDir,filesOut[i])
    out=" ".join(filesOut)
    #print("Echo Files included: {0}".format(out))
    
    if isOutStr:
        return out
    else:
        return filesOut
    
    


def loadFile(file, datasetname=None):

    if file[-4:]==".txt":
        data=np.array(np.loadtxt(file, dtype=np.float32))
        print("Shape: {0}".format(np.shape(data)))
        return data
    if file[-4:] == ".nii" or file[-7:] == ".nii.gz":
        #data=nli.load_img(file).get_fdata()
        data=nli.load_img(file)
        #print("Shape: {0}".format(np.shape(data)))
        print("Shape: {0}".format(data.shape))
        return data
    if file[-4:] == ".tsv":
        df = pd.read_csv(file, sep="\t")
        data=np.array(df.values)
        print("Shape: {0}".format(np.shape(data)))
        data=data.astype('float32')
        return data
    if file[-4:]==".npy":
        data=np.astype(np.array(np.load(file)), np.float32) 
        print("Shape: {0}".format(np.shape(data)))
        return data
        
    import_h5py()
    if datasetname is None:
        datasetname="data"
    with h5py.File(file, "r") as h5f:
        dataset = h5f[datasetname]
        data=np.array(dataset[()])
        print("Shape: {0}".format(np.shape(data)))
        data=data.astype('float32')
        return np.array(dataset[()])
        
    assert 1==0, "Error loading file {0}".format(file)

def fileSortHelperType2(fileName):
    ## Handles .nii.gz and .XXX extensions (including .nii) only 

    if fileName[-7:] == ".nii.gz":
        name = fileName[:-7]
    else:
        name = fileName[:-4]
    start = name.index("_echo-")
    start += 6
    end=start+1
    index=name[start:end]
    
    
    while index.isnumeric():
        end += 1
        index = name[start:end]
    if not index.isnumeric():
        index = name[start:end-1]
        if not index.isnumeric():
            raise Exception("Incorrect file format. fileSortHelper handles .nii.gz and .XXX extensions only")
    
    return int(index)
    

def runFullTedana(echoDir,savedir,tes,brain_mask_path, save_prefix,echo_dof=None,echosToDownload=None, tes_conversion_factor=10**3,
                  echofilePattern="*_echo-*_desc-preproc_bold.nii*",echofileSortType=2,topOfBidsPath=None,
                  mixm=None, tree="tedana_orig",overwrite=False):

        
    
    if isinstance(tes, str):
        assert tes[-4:] == ".txt", "tes must be an array or a .txt file"
        with contextlib.redirect_stdout(None):
            tes=loadFile(tes)

    echoFiles=strOfFilePaths(echoDir,echosToDownload,echofilePattern,echofileSortType,isOutStr=0)
    tes=np.array(tes)*tes_conversion_factor
    if echosToDownload is not None:
        echosToDownload=[i-1 for i in echosToDownload]
        tes=tes[echosToDownload]
    else:
        echoesToDownload=None

    
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)

    tedana_update.tedana_workflow(
        echoFiles,
        tes,
        out_dir=savedir,
        mask=brain_mask_path,
        convention="bids",
        prefix=save_prefix,
        echo_dof=echo_dof,
        tree=tree,
        external_regressors=None,
        verbose=True,
        overwrite=overwrite,
        mixing_file=mixm,
    )

    

    if topOfBidsPath is not None and not os.path.exists(os.path.join(topOfBidsPath,"dataset_description.json")):
        json_file=os.path.join(topOfBidsPath,"dataset_description.json")
        cmd = "touch {0}".format(json_file)
        os.system(cmd)
        dictionary={
          "Name": "Tedana",
          "BIDSVersion": "1.4.0",
          "DatasetType": "derivative"
        }
        with open(json_file, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
            json_file.close() 

        files_to_remove = glob.glob(os.path.join(savedir,"*dataset_description.json"))
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
            except:
                print("Unable to remove: {0}".format(file_path))

def subsampleEchoes(echoDir,savedir,tes,brain_mask_path, 
                    save_prefix,echo_dof=None,echo_start=None,echo_stop=None,echo_step_size=None,echoes=None,
                    tes_conversion_factor=10**3,echofilePattern="*_echo-*_desc-preproc_bold.nii*",echofileSortType=2,
                    topOfBidsPath=None,mixm=None, tree="tedana_orig",overwrite=False):

    
    if echoes:
        echosToDownload=echoes
    elif echo_start and echo_stop and echo_step_size:
        echo_step_size=max(int(echo_step_size),1)
        echosToDownload=np.arange(echo_start,echo_stop+1,echo_step_size)
    else:
        assert False, "echo_start, echo_stop, and echo_step_size must be numeric OR echoes must be a nonempty list."

    
    runFullTedana(echoDir,savedir,tes,brain_mask_path, save_prefix,echo_dof=echo_dof, 
                  echosToDownload=echosToDownload,tes_conversion_factor=tes_conversion_factor,
                  echofilePattern=echofilePattern,echofileSortType=echofileSortType,
                  topOfBidsPath=topOfBidsPath,mixm=mixm,tree=tree,overwrite=overwrite)


## Example: 
# subsampleEchoes(echoDir="/data/cfmri_users/data/070524/preprocessed/fmriprep24.0.1/sub-01/ses-EPTItap/func",
#                 savedir="/data/cfmri_users/data/070524/derivatives/tedana24.0.2_preproc-fmriprep24.0.1_desc-61echoes/sub-01/ses-EPTItap61/func",
#                 tes="/data/cfmri_users/data/070524/TEs/sub-01_ses-EPTItap_TEs.txt",
#                 brain_mask_path="/data/cfmri_users/data/070524/preprocessed/fmriprep24.0.1/sub-01/ses-EPTItap/func/sub-01_ses-EPTItap_task-tap_desc-brain_mask.nii.gz", 
#                 save_prefix="sub-01_ses-EPTItap61_task-tap", 
#                 echo_start=1,
#                 echo_stop=122,
#                 echo_step_size=2,
#                 topOfBidsPath="/data/cfmri_users/data/070524/derivatives/tedana24.0.2_preproc-fmriprep24.0.1_desc-61echoes")
    



    
        
