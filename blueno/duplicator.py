import shutil
num_copies = 330

for i in range(1,num_copies):
    dest = "blueno_" + str(i) + ".off"
    shutil.copy("blueno_0.off", dest)