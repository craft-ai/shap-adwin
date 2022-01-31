import os

dirList = os.listdir("./results") # current directory
print(dirList)
for dir in dirList:
  if os.path.isdir(dir) == True:
    print("A",dir)
    # I don't know how to get into this dir and do the same thing here
  else:
    print("else", dir)


    # I got file and i can regexp if it is .htm|html
