import os,glob

dirc = os.getcwd()
count=0
for f in glob.glob(os.path.join(dirc,"*.jpg")):
    
    os.rename(f,'lol_'+str(count)+'.jpg')
    count+=1