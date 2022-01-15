import shutil
# num_copies = 330

# for i in range(1,num_copies):
#     dest = "blueno_" + str(i) + ".off"
#     shutil.copy("blueno_0.off", dest)

file = open("./960generator.txt", "w+")
s = ""
for i in range(960):
    s += "ECHO " + str(i) + "/960\n"
    s += '"C:\Program Files\OpenSCAD\openscad.exe" -o blueno' + str(i) + ".stl blueno.scad\n"
print(s)
file.write(s)
file.close()